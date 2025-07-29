from pathlib import Path
import os
import argparse
import json
import random
import numpy as np
from dataclasses import dataclass, field
import wandb

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperTokenizerFast, WhisperConfig, AdamW
from tqdm import tqdm
import evaluate
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from whistress.inference_client.utils import prepare_audio, save_model_parts, get_loaded_model
from whistress.model.model import WhiStress

precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_prf_metrics(predictions, references, average="binary"):
    """
    Computes precision, recall, and F1 using Hugging Face's `evaluate`.
    Args:
        predictions (List[int]): Model's predicted labels.
        references  (List[int]): True labels.
        average     (str): "binary", "macro", "micro", or "weighted".
                          Use "binary" for two-class tasks.
    Returns:
        Dict[str, float]: e.g. {"precision": 0.8, "recall": 0.75, "f1": 0.77}
    """
    p = precision_metric.compute(predictions=predictions, references=references, average=average)["precision"]
    r = recall_metric.compute(predictions=predictions, references=references, average=average)["recall"]
    f = f1_metric.compute(predictions=predictions, references=references, average=average)["f1"]

    return {"precision": p, "recall": r, "f1": f}

def compute_stress_binary(transcription: str, emphasis_indices: list[int]) -> list[int]:
    words = transcription.strip().split()
    return [1 if i in emphasis_indices else 0 for i in range(len(words))]

def preprocess(example, model, feats_aug="default"):
    # 1. word-level binary label
    binary = compute_stress_binary(example["transcription"], example["emphasis_indices"])
    example["stress_pattern"] = {"binary": binary}

    # 2. prepare audio
    array = prepare_audio(audio=example["audio"])
    example["audio_input"] = {
        "array": array,
        "sampling_rate": 16000
    }

    # 3. tokenize transcription
    transcription = example["transcription"]
    tokenized = model.processor.tokenizer(
        transcription,
        return_tensors="pt",
        truncation=True,
        max_length=50
    )

    input_ids = tokenized["input_ids"][0]
    decoded_tokens = [
        model.processor.tokenizer.decode([tid], skip_special_tokens=False)
        for tid in input_ids
    ]

    # 4. align word-level binary to token-level
    token_labels = []
    word_ids = []
    word_idx = 0
    first_real_token = True
    for token in decoded_tokens:
        if token in model.processor.tokenizer.all_special_tokens:
            token_labels.append(-100)
            continue

        if token.startswith(" ") or first_real_token:  # new word
            label = binary[word_idx] if word_idx < len(binary) else 0
            token_labels.append(label)
            word_idx += 1
            first_real_token = False
            word_ids.append(word_idx)
        else:  # subword
            label = binary[word_idx - 1] if word_idx - 1 < len(binary) else 0
            token_labels.append(label)
            word_ids.append(word_idx -1)

    # 5. LEFT SHIFT → align label with logits[t] = prediction of token[t+1]
    token_labels = token_labels[1:] + [-100]

    # 6. store results
    example["decoder_input_ids"] = input_ids
    example["labels_head"] = token_labels
    example["word_ids"] = word_ids
    
    # 7. feats_aug
    if feats_aug == "raw_feats":
        pass
    
    return example

class StressDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, model, feats_aug):
        self.feats_aug = feats_aug
        self.dataset = hf_dataset.map(lambda x: preprocess(x, model=model, feats_aug=self.feats_aug), num_proc=1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        if self.feats_aug == "default":
            return {
                "audio": item["audio"],
                "audio_input": item["audio_input"],
                "decoder_input_ids": item["decoder_input_ids"],
                "labels_head": item["labels_head"],
                "transcription": item["transcription"],
                "stress_pattern_binary": item["stress_pattern"]["binary"],
                "word_ids": item["word_ids"],
                "id": item["id"]
            }
        elif self.feats_aug == "raw_feats":
            return {
                "audio": item["audio"],
                "audio_input": item["audio_input"],
                "decoder_input_ids": item["decoder_input_ids"],
                "labels_head": item["labels_head"],
                "transcription": item["transcription"],
                "stress_pattern_binary": item["stress_pattern"]["binary"],
                "word_ids": item["word_ids"],
                "raw_feats": item["raw_feats"],
                "id": item["id"]
            }

@dataclass
class MyCollate:
    processor: WhisperProcessor
    feats_aug: "default"

    def __call__(self, batch):
        
        if self.feats_aug == "default":
            decoder_input_ids = [torch.tensor(b["decoder_input_ids"]) for b in batch]
            word_ids = [torch.tensor(b["word_ids"]) for b in batch]
            labels_head = [torch.tensor(b["labels_head"]) for b in batch]
            
            return {
                "audio": [b["audio"] for b in batch],
                "audio_input": [b["audio_input"] for b in batch],
                "decoder_input_ids": pad_sequence(decoder_input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id),
                "word_ids": pad_sequence(word_ids, batch_first=True, padding_value=-100),
                "labels_head": pad_sequence(labels_head, batch_first=True, padding_value=-100),
                "transcription": [b["transcription"] for b in batch],
                "stress_pattern_binary": [b["stress_pattern_binary"] for b in batch],
                "id": [b["id"] for b in batch]
            }
        elif self.feats_aug == "raw_feats":
            decoder_input_ids = [torch.tensor(b["decoder_input_ids"]) for b in batch]
            labels_head = [torch.tensor(b["labels_head"]) for b in batch]
            raw_feats = [torch.tensor(b["raw_feats"]) for b in batch]
            
            return {
                "audio": [b["audio"] for b in batch],
                "audio_input": [b["audio_input"] for b in batch],
                "decoder_input_ids": pad_sequence(decoder_input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id),
                "word_ids": pad_sequence(word_ids, batch_first=True, padding_value=-100),
                "labels_head": pad_sequence(labels_head, batch_first=True, padding_value=-100),
                "transcription": [b["transcription"] for b in batch],
                "stress_pattern_binary": [b["stress_pattern_binary"] for b in batch],
                "raw_feats": pad_sequence(raw_feats, batch_first=True, padding_value=0),
                "id": [b["id"] for b in batch]
            }
