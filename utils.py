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
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from whistress.inference_client.utils import prepare_audio, save_model_parts, get_loaded_model
from whistress.model.model import WhiStress

import os
from pathlib import Path
import re
import nltk
nltk.download('cmudict')
from nltk.corpus import cmudict

cmu = cmudict.dict()

def build_phone2id_no_stress(path="data/local/phones.txt"):
    path = Path(path)
    if path.exists():
        # read mapping
        phone2id = {}
        with open(path, "r") as f:
            for line in f:
                phone, pid = line.strip().split()
                phone2id[phone] = int(pid)
        return phone2id
    else:
        # create directory if not exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # build from cmudict
        all_phones = set()
        for word in cmu:
            for pron in cmu[word]:
                for ph in pron:
                    ph_no_stress = re.sub(r"\d", "", ph)
                    all_phones.add(ph_no_stress)

        phone_list = sorted(list(all_phones))
        phone2id = {ph: idx for idx, ph in enumerate(phone_list)}

        # save
        with open(path, "w") as f:
            for ph, idx in phone2id.items():
                f.write(f"{ph} {idx}\n")

        return phone2id

def add_phone_features(transcription, phone_dict):
    phones = []
    phone_ids = []
    phone_stress = []
    offset = 0
    
    for word in transcription.lower().split():
        phone_list = cmu.get(word)
        
        if not phone_list:
            continue
        
        phone_list = phone_list[0]
        
        for phn in phone_list:
            phones.append(phn)
            phn_no_stress = re.sub(r"\d", "", phn)
            phone_ids.append(phone_dict[phn_no_stress])
            
            if phn[-1].isdigit():
                phone_stress.append(int(phn[-1]))
            else:
                phone_stress.append(0)
    
    return phones, phone_ids, phone_stress

def compute_stress_binary(transcription: str, emphasis_indices: list[int]) -> list[int]:
    words = transcription.strip().split()
    return [1 if i in emphasis_indices else 0 for i in range(len(words))]

def preprocess(example, model, phone_dict):
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
    
    # 6. Get phone sequence from transcription
    phones, phone_ids, phone_stress = add_phone_features(transcription, phone_dict)

    # 7. store results
    example["decoder_input_ids"] = input_ids
    example["labels_head"] = token_labels
    example["word_ids"] = word_ids
    example["phones"] = phones
    example["phone_ids"] = phone_ids
    example["phone_stress"] = phone_stress
    
    return example

class StressDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, model):
        self.phone_dict = build_phone2id_no_stress()
        self.dataset = hf_dataset.map(lambda x: preprocess(x, model=model, phone_dict=self.phone_dict), num_proc=1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        return {
            "audio": item["audio"],
            "audio_input": item["audio_input"],
            "decoder_input_ids": item["decoder_input_ids"],
            "labels_head": item["labels_head"],
            "transcription": item["transcription"],
            "stress_pattern_binary": item["stress_pattern"]["binary"],
            "word_ids": item["word_ids"],
            "phones": item["phones"],
            "phone_ids": item["phone_ids"],
            "phone_stress": item["phone_stress"],
            "id": item["id"]
        }

@dataclass
class MyCollate:
    processor: WhisperProcessor

    def __call__(self, batch):
        
        decoder_input_ids = [torch.tensor(b["decoder_input_ids"]) for b in batch]
        word_ids = [torch.tensor(b["word_ids"]) for b in batch]
        labels_head = [torch.tensor(b["labels_head"]) for b in batch]
        phone_ids = [torch.tensor(b["phone_ids"]) for b in batch]
        phone_stress = [torch.tensor(b["phone_stress"]) for b in batch]
            
        return {
            "audio": [b["audio"] for b in batch],
            "audio_input": [b["audio_input"] for b in batch],
            "decoder_input_ids": pad_sequence(decoder_input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id),
            "word_ids": pad_sequence(word_ids, batch_first=True, padding_value=-100),
            "labels_head": pad_sequence(labels_head, batch_first=True, padding_value=-100),
            "phone_ids": pad_sequence(phone_ids, batch_first=True, padding_value=-100),
            "phone_stress": pad_sequence(phone_stress, batch_first=True, padding_value=-100),
            "transcription": [b["transcription"] for b in batch],
            "stress_pattern_binary": [b["stress_pattern_binary"] for b in batch],
            "phones": [b["phones"] for b in batch],
            "id": [b["id"] for b in batch]
        }

if __name__ == "__main__":
    phone_dict = build_phone2id_no_stress()
    phones, phone_ids, phone_stress = add_phone_features("Nice to meet you", phone_dict)
    print(phone_dict)
    print(phones)
    print(phone_ids)
    print(phone_stress)
