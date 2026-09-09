import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StressRealizationCoupling(nn.Module):
    """Couple lexical stress location with sentence-level realization strength.

    WSD identifies the primary-stress vowel within each word. SSD estimates
    how strongly that lexical locus is realized relative to other words in
    the utterance. Preliminary predictions refine each other through a shared
    vowel-level realization score without exchanging raw branch embeddings.
    """

    def __init__(
        self,
        d_model,
        temperature=1.0,
        margin=0.2,
        sentence_to_lexical_gate_init=0.01,
        lexical_to_sentence_gate_init=0.01,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("realization temperature must be positive")

        self.temperature = float(temperature)
        self.margin = float(margin)
        self.realization_norm = nn.LayerNorm(d_model)
        self.realization_head = nn.Linear(d_model, 1)
        self.sentence_to_lexical_gate_logit = nn.Parameter(
            torch.tensor(self._gate_logit(sentence_to_lexical_gate_init))
        )
        self.lexical_to_sentence_gate_logit = nn.Parameter(
            torch.tensor(self._gate_logit(lexical_to_sentence_gate_init))
        )

    @staticmethod
    def _gate_logit(value):
        value = float(value)
        if not 0.0 < value < 1.0:
            raise ValueError("realization gate initial values must be in (0, 1)")
        return math.log(value / (1.0 - value))

    @staticmethod
    def _binary_logit_delta(margin_delta):
        """Convert a margin change into a centered two-class logit change."""
        return torch.stack((-0.5 * margin_delta, 0.5 * margin_delta), dim=-1)

    def forward(
        self,
        ssd_hidden_states,
        phone_hidden_states,
        preliminary_ssd_logits,
        preliminary_phone_logits,
        word_ids,
        phone_word_ids,
        phone_vowel_mask,
        labels_head=None,
    ):
        if ssd_hidden_states.shape[:2] != word_ids.shape:
            raise ValueError("ssd_hidden_states and word_ids must align on [B, T]")
        if phone_hidden_states.shape[:2] != phone_word_ids.shape:
            raise ValueError(
                "phone_hidden_states and phone_word_ids must align on [B, T_phone]"
            )
        if preliminary_ssd_logits.shape[:2] != word_ids.shape:
            raise ValueError(
                "preliminary_ssd_logits and word_ids must align on [B, T]"
            )
        if preliminary_phone_logits.shape[:2] != phone_word_ids.shape:
            raise ValueError(
                "preliminary_phone_logits and phone_word_ids must align on "
                "[B, T_phone]"
            )
        if phone_word_ids.shape != phone_vowel_mask.shape:
            raise ValueError(
                "phone_word_ids and phone_vowel_mask must have identical shape"
            )
        if labels_head is not None and labels_head.shape != word_ids.shape:
            raise ValueError("labels_head and word_ids must have identical shape")

        realization_scores = torch.tanh(
            self.realization_head(
                self.realization_norm(phone_hidden_states)
            ).squeeze(-1)
        )
        sentence_to_lexical_gate = torch.sigmoid(
            self.sentence_to_lexical_gate_logit
        )
        lexical_to_sentence_gate = torch.sigmoid(
            self.lexical_to_sentence_gate_logit
        )
        preliminary_ssd_margins = (
            preliminary_ssd_logits[..., 1]
            - preliminary_ssd_logits[..., 0]
        )
        preliminary_phone_margins = (
            preliminary_phone_logits[..., 1]
            - preliminary_phone_logits[..., 0]
        )

        ssd_logit_delta = torch.zeros_like(preliminary_ssd_logits)
        phone_logit_delta = torch.zeros_like(preliminary_phone_logits)
        realization_ranking_terms = []

        for b in range(ssd_hidden_states.size(0)):
            word_records = []
            valid_word_ids = torch.unique(word_ids[b][word_ids[b] >= 0])

            for wid in valid_word_ids:
                token_mask = word_ids[b] == wid
                vowel_mask = (
                    (phone_word_ids[b] == wid)
                    & phone_vowel_mask[b].bool()
                )
                if not token_mask.any() or not vowel_mask.any():
                    continue

                # SSD -> WSD: sentence prominence controls how strongly the
                # within-word vowel contrast refines lexical-stress logits.
                word_ssd_probability = torch.sigmoid(
                    preliminary_ssd_margins[b][token_mask].mean()
                )
                word_vowel_scores = realization_scores[b][vowel_mask]
                centered_vowel_scores = (
                    word_vowel_scores - word_vowel_scores.mean()
                )
                phone_margin_delta = (
                    sentence_to_lexical_gate
                    * word_ssd_probability
                    * centered_vowel_scores
                )
                phone_logit_delta[b, vowel_mask] = self._binary_logit_delta(
                    phone_margin_delta
                )

                # WSD -> SSD: the refined WSD posterior selects the vowel
                # whose realized prominence should represent this word.
                refined_vowel_margins = (
                    preliminary_phone_margins[b][vowel_mask]
                    + phone_margin_delta
                )
                lexical_locus_weights = F.softmax(
                    refined_vowel_margins / self.temperature,
                    dim=0,
                )
                word_realization = torch.sum(
                    lexical_locus_weights * word_vowel_scores
                )

                word_label = None
                if labels_head is not None:
                    valid_labels = labels_head[b][token_mask]
                    valid_labels = valid_labels[valid_labels != -100]
                    if valid_labels.numel() > 0:
                        word_label = int(valid_labels.max().item())

                word_records.append(
                    (token_mask, word_realization, word_label)
                )

            if not word_records:
                continue

            # Sentence prominence is relative. Center each lexical-locus
            # realization against the other aligned words in the utterance.
            utterance_realizations = torch.stack(
                [record[1] for record in word_records]
            )
            utterance_center = utterance_realizations.mean()

            for token_mask, word_realization, _ in word_records:
                ssd_margin_delta = (
                    lexical_to_sentence_gate
                    * (word_realization - utterance_center)
                )
                ssd_logit_delta[b, token_mask] = self._binary_logit_delta(
                    ssd_margin_delta
                )

            if labels_head is not None:
                positive_realizations = [
                    record[1] for record in word_records if record[2] == 1
                ]
                negative_realizations = [
                    record[1] for record in word_records if record[2] == 0
                ]
                for positive in positive_realizations:
                    for negative in negative_realizations:
                        realization_ranking_terms.append(
                            F.softplus(self.margin - positive + negative)
                        )

        if labels_head is None:
            loss_realization = None
        elif realization_ranking_terms:
            loss_realization = torch.stack(realization_ranking_terms).mean()
        else:
            loss_realization = realization_scores.sum() * 0.0

        return ssd_logit_delta, phone_logit_delta, loss_realization
