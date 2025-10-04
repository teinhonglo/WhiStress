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
from datasets import load_dataset, load_from_disk
from transformers import WhisperProcessor, WhisperTokenizerFast, WhisperConfig, AdamW
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from whistress.inference_client.utils import prepare_audio, save_model_parts, get_loaded_model
from whistress.model.model import WhiStress

import os
from pathlib import Path
import re
import string
import nltk
nltk.download('cmudict')
from nltk.corpus import cmudict
cmu = cmudict.dict()

from g2p_en import G2p
from local.e2e_stt.nlp_models import NlpModel

POS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", 
       "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", 
       "PUNCT", "SCONJ", "SYM", "VERB", "X"]
pos_map_dict = {p: i for i, p in enumerate(POS)}

g2p = G2p()
nlp_model = NlpModel(tokenize_pretokenized=True)

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
        for word, phone_list in cmudict.dict().items():
            for phones in phone_list:
                for ph in phones:
                    ph = re.sub(r'\d$', '', ph)
                    all_phones.add(ph)
        
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
    transcription = transcription.translate(str.maketrans('', '', string.punctuation))
    
    transcription_phones = []
    word_phones = []
    for phone in g2p(transcription):
        if phone == " ":
            transcription_phones.append([p for p in word_phones])
            word_phones = []
            continue
        
        word_phones.append(phone)
    transcription_phones.append([p for p in word_phones])
    
    words = transcription.lower().split()
    assert len(words) == len(transcription_phones)
    
    for phone_list in transcription_phones:
        
        for phn in phone_list:
            phones.append(phn)
            phn_no_stress = re.sub(r"\d", "", phn)
            phone_ids.append(phone_dict[phn_no_stress])
            
            if phn[-1].isdigit():
                stress_label = 1 if int(phn[-1]) > 0 else 0
                phone_stress.append(stress_label)
            else:
                phone_stress.append(0)
        if len(phone_ids) == 0:
            print(transcription)

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
    
    vp_feats = nlp_model.vocab_profile_feats(transcription.split())
    pos_feats = vp_feats["pos_list"]

    # 4. align word-level binary to token-level
    token_labels = []
    token_pos = []
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
            token_pos.append(pos_map_dict[pos_feats[word_idx]])
            word_ids.append(word_idx)
            word_idx += 1
            first_real_token = False
        else:  # subword
            label = binary[word_idx - 1] if word_idx - 1 < len(binary) else 0
            token_labels.append(label)
            token_pos.append(pos_map_dict[pos_feats[word_idx -1]])
            word_ids.append(word_idx -1)

    # 5. LEFT SHIFT → align label with logits[t] = prediction of token[t+1]
    token_labels = token_labels[1:] + [-100]
    
    # 6. Get phone sequence from transcription
    phones, phone_ids, phone_labels_head = add_phone_features(transcription, phone_dict)

    # 7. store results
    example["decoder_input_ids"] = input_ids
    example["labels_head"] = token_labels
    example["word_ids"] = word_ids
    example["token_pos"] = token_pos
    example["phones"] = phones
    example["phone_ids"] = phone_ids
    example["phone_labels_head"] = phone_labels_head
    
    return example

class StressDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset_or_path, model, processed_dir="data/processed", num_proc=1):
        self.phone_dict = build_phone2id_no_stress()

        # 如果有快取就直接讀
        if os.path.exists(processed_dir):
            self.dataset = load_from_disk(processed_dir)
        else:
            # 如果給的是路徑 → 先讀 raw dataset
            if isinstance(hf_dataset_or_path, str):
                hf_dataset = load_from_disk(hf_dataset_or_path)
            else:
                hf_dataset = hf_dataset_or_path

            # 做 map
            self.dataset = hf_dataset.map(
                lambda x: preprocess(x, model=model, phone_dict=self.phone_dict),
                num_proc=num_proc
            )
            # 存快取
            self.dataset.save_to_disk(processed_dir)

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
            "token_pos": item["token_pos"],
            "phones": item["phones"],
            "phone_ids": item["phone_ids"],
            "phone_labels_head": item["phone_labels_head"],
            "id": item["id"]
        }

@dataclass
class MyCollate:
    processor: WhisperProcessor

    def __call__(self, batch):
        
        decoder_input_ids = [torch.tensor(b["decoder_input_ids"], dtype=torch.long) for b in batch]
        word_ids = [torch.tensor(b["word_ids"], dtype=torch.long) for b in batch]
        token_pos = [torch.tensor(b["token_pos"], dtype=torch.long) for b in batch]
        labels_head = [torch.tensor(b["labels_head"], dtype=torch.long) for b in batch]
        phone_ids = [torch.tensor(b["phone_ids"], dtype=torch.long) for b in batch]
        phone_labels_head = [torch.tensor(b["phone_labels_head"], dtype=torch.long) for b in batch]
            
        return {
            "audio": [b["audio"] for b in batch],
            "audio_input": [b["audio_input"] for b in batch],
            "decoder_input_ids": pad_sequence(decoder_input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id),
            "word_ids": pad_sequence(word_ids, batch_first=True, padding_value=-100),
            "token_pos": pad_sequence(token_pos, batch_first=True, padding_value=-1),
            "labels_head": pad_sequence(labels_head, batch_first=True, padding_value=-100),
            "phone_ids": pad_sequence(phone_ids, batch_first=True, padding_value=-1),
            "phone_labels_head": pad_sequence(phone_labels_head, batch_first=True, padding_value=-100),
            "transcription": [b["transcription"] for b in batch],
            "stress_pattern_binary": [b["stress_pattern_binary"] for b in batch],
            "phones": [b["phones"] for b in batch],
            "id": [b["id"] for b in batch]
        }

if __name__ == "__main__":
    phone_dict = build_phone2id_no_stress()
    phones, phone_ids, phone_stress = add_phone_features("Nice to meet you.", phone_dict)
    print(phone_dict)
    print(phones)
    print(phone_ids)
    print(phone_stress)
