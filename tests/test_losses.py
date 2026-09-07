import torch

from losses import (
    compute_cross_granularity_conditional_ranking_loss,
    compute_word_level_mil_loss,
)


def test_word_level_mil_rewards_one_positive_subtoken():
    labels = torch.tensor([[1, 1]], dtype=torch.long)
    word_ids = torch.tensor([[0, 0]], dtype=torch.long)

    logits_good = torch.tensor(
        [[[0.0, 4.0], [4.0, 0.0]]], dtype=torch.float32
    )
    logits_bad = torch.tensor(
        [[[4.0, 0.0], [4.0, 0.0]]], dtype=torch.float32
    )

    good = compute_word_level_mil_loss(
        logits_good, labels, word_ids, temperature=0.25
    )
    bad = compute_word_level_mil_loss(
        logits_bad, labels, word_ids, temperature=0.25
    )

    assert good < bad


def test_conditional_ranking_prefers_primary_alignment():
    labels = torch.tensor([[1]], dtype=torch.long)
    word_ids = torch.tensor([[0]], dtype=torch.long)
    phone_labels = torch.tensor([[1, 0]], dtype=torch.long)
    phone_word_ids = torch.tensor([[0, 0]], dtype=torch.long)
    vowel_mask = torch.tensor([[1, 1]], dtype=torch.long)

    ssd = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
    phone_good = torch.tensor(
        [[[1.0, 0.0], [-1.0, 0.0]]], dtype=torch.float32
    )
    phone_bad = torch.tensor(
        [[[-1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32
    )

    good = compute_cross_granularity_conditional_ranking_loss(
        ssd,
        phone_good,
        labels,
        word_ids,
        phone_labels,
        phone_word_ids,
        vowel_mask,
        margin=0.2,
    )
    bad = compute_cross_granularity_conditional_ranking_loss(
        ssd,
        phone_bad,
        labels,
        word_ids,
        phone_labels,
        phone_word_ids,
        vowel_mask,
        margin=0.2,
    )

    assert good < bad


def test_conditional_ranking_skips_unstressed_words():
    labels = torch.tensor([[0]], dtype=torch.long)
    word_ids = torch.tensor([[0]], dtype=torch.long)
    phone_labels = torch.tensor([[1, 0]], dtype=torch.long)
    phone_word_ids = torch.tensor([[0, 0]], dtype=torch.long)
    vowel_mask = torch.tensor([[1, 1]], dtype=torch.long)
    ssd = torch.randn(1, 1, 4, requires_grad=True)
    phone = torch.randn(1, 2, 4, requires_grad=True)

    loss = compute_cross_granularity_conditional_ranking_loss(
        ssd,
        phone,
        labels,
        word_ids,
        phone_labels,
        phone_word_ids,
        vowel_mask,
    )

    assert loss.item() == 0.0
