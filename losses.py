import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from typing import Dict, List, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.modules.loss import _Loss
import os
import ast
import logging
from collections import Counter, defaultdict
from torch.autograd import Variable

def compute_adaptive_weighted_loss(logits, labels_head, word_ids):
    B, T, _ = logits.shape
    probs = F.softmax(logits, dim=-1)[..., 1]  # (B, T) — probability of class 1 (stressed) per token

    total_loss = 0.0
    count = 0

    for b in range(B):  # loop over each sample in the batch
        word2token = defaultdict(list)

        # Step 0: Group token indices by word ID (excluding special tokens and paddings)
        for t, wid in enumerate(word_ids[b]):
            if wid != -100 and labels_head[b, t] != -100 and labels_head[b, t] == 1:
                word2token[wid].append(t)

        # Step 1–3: For each word/syllable group
        for token_indices in word2token.values():
            p = probs[b, token_indices]  # get predicted probabilities for all tokens in the word

            if len(p) == 0:
                continue  # skip if no valid token

            # === Step 1: Adaptive Weight ===
            # ω_α = |1 - ∑p_i| — how much the total predicted stress deviates from 1
            omega = torch.abs(1.0 - p.sum())

            # === Step 2: Log Penalty Term ===
            # Identify the token with highest predicted stress
            i_max = torch.argmax(p)

            # First term: log(p_{i_max})
            penalty = torch.log(p[i_max] + 1e-8)  # add small constant to avoid log(0)

            # Remaining terms: sum log(1 - p_i) for i ≠ i_max
            for j, p_j in enumerate(p):
                if j != i_max:
                    penalty += torch.log(1.0 - p_j + 1e-8)

            # === Step 3: Final WP = ω_α * penalty
            word_loss = omega * penalty

            total_loss += word_loss
            count += 1

    # Return mean loss over valid words (or 0.0 if none)
    return -1 * total_loss / count if count > 0 else torch.tensor(0.0, device=logits.device)


def compute_word_level_mil_loss(
    logits,
    labels_head,
    word_ids,
    temperature=1.0,
):
    """Word-level MIL loss for token-level SSD logits.

    Token logits within the same lexical word are aggregated with a
    temperature-controlled log-mean-exp over the stressed-vs-unstressed
    logit margin. This approaches max pooling as the temperature decreases
    while avoiding a bias toward words with more subword tokens.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if logits.shape[:2] != labels_head.shape or labels_head.shape != word_ids.shape:
        raise ValueError(
            "logits, labels_head, and word_ids must share [B, T] dimensions"
        )

    stress_margin = logits[..., 1] - logits[..., 0]
    losses = []

    for b in range(logits.size(0)):
        valid_word_ids = torch.unique(word_ids[b][word_ids[b] >= 0])
        for wid in valid_word_ids:
            token_mask = (word_ids[b] == wid) & (labels_head[b] != -100)
            if not token_mask.any():
                continue

            token_margins = stress_margin[b][token_mask]
            n_tokens = token_margins.numel()
            word_logit = temperature * (
                torch.logsumexp(token_margins / temperature, dim=0)
                - math.log(n_tokens)
            )

            token_labels = labels_head[b][token_mask]
            # SSD labels are replicated across subtokens. max() is robust to
            # malformed mixed labels while preserving the word-level target.
            word_label = token_labels.max().to(dtype=word_logit.dtype)
            losses.append(
                F.binary_cross_entropy_with_logits(word_logit, word_label)
            )

    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def compute_cross_granularity_conditional_ranking_loss(
    ssd_hidden_states,
    phone_hidden_states,
    labels_head,
    word_ids,
    phone_labels_head,
    phone_word_ids,
    phone_vowel_mask,
    margin=0.2,
):
    """Relate sentence prominence to the lexical primary-stress locus.

    For a sentence-stressed word, its pooled SSD representation should be
    more similar to the representation of the primary-stressed vowel than
    to the non-primary vowel representation from the same word.

    The constraint is intentionally one-sided: an unstressed sentence word
    still retains lexical stress, so SSD=0 does not imply the opposite
    ranking. Words without both primary and non-primary vowel candidates are
    skipped (e.g. most monosyllabic words).
    """
    if ssd_hidden_states.shape[:2] != labels_head.shape:
        raise ValueError("ssd_hidden_states and labels_head must align on [B, T]")
    if labels_head.shape != word_ids.shape:
        raise ValueError("labels_head and word_ids must have identical shape")
    if phone_hidden_states.shape[:2] != phone_labels_head.shape:
        raise ValueError(
            "phone_hidden_states and phone_labels_head must align on [B, T_phone]"
        )
    if phone_labels_head.shape != phone_word_ids.shape:
        raise ValueError(
            "phone_labels_head and phone_word_ids must have identical shape"
        )
    if phone_labels_head.shape != phone_vowel_mask.shape:
        raise ValueError(
            "phone_labels_head and phone_vowel_mask must have identical shape"
        )

    losses = []

    for b in range(ssd_hidden_states.size(0)):
        valid_word_ids = torch.unique(word_ids[b][word_ids[b] >= 0])
        for wid in valid_word_ids:
            token_mask = (word_ids[b] == wid) & (labels_head[b] != -100)
            if not token_mask.any():
                continue

            # Apply the cross-granularity relation only when the word is
            # sentence-stressed. Lexical stress remains present when SSD=0.
            word_label = labels_head[b][token_mask].max()
            if word_label.item() != 1:
                continue

            phone_mask = (
                (phone_word_ids[b] == wid)
                & (phone_labels_head[b] != -100)
                & phone_vowel_mask[b].bool()
            )
            primary_mask = phone_mask & (phone_labels_head[b] == 1)
            nonprimary_mask = phone_mask & (phone_labels_head[b] == 0)

            if not primary_mask.any() or not nonprimary_mask.any():
                continue

            ssd_word = ssd_hidden_states[b][token_mask].mean(dim=0)
            primary_repr = phone_hidden_states[b][primary_mask].mean(dim=0)
            nonprimary_repr = phone_hidden_states[b][nonprimary_mask].mean(dim=0)

            sim_primary = F.cosine_similarity(
                ssd_word.unsqueeze(0), primary_repr.unsqueeze(0), dim=-1
            ).squeeze(0)
            sim_nonprimary = F.cosine_similarity(
                ssd_word.unsqueeze(0), nonprimary_repr.unsqueeze(0), dim=-1
            ).squeeze(0)

            losses.append(
                F.softplus(margin - sim_primary + sim_nonprimary)
            )

    if not losses:
        return ssd_hidden_states.sum() * 0.0
    return torch.stack(losses).mean()

class ComputeLoss(nn.Module):
    def __init__(self, model_args, class_weights=None):
        super(ComputeLoss, self).__init__()
        self.loss_type = model_args["loss_type"]
        self.class_weights = class_weights
        if self.class_weights:
            self.loss_fct = CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
        else:
            self.loss_fct = CrossEntropyLoss(ignore_index=-100)

    def forward(self, logits, labels, hidden_states=None, word_ids=None):
        if self.loss_type == "default":
            loss = self.loss_fct(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        return loss, logits
