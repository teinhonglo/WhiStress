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
