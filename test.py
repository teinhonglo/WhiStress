from pathlib import Path
import os
import argparse
import json
import random
import numpy as np
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, WhisperTokenizerFast, WhisperConfig, AdamW
from tqdm import tqdm
import evaluate
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from whistress.inference_client.utils import prepare_audio, save_model_parts, get_loaded_model
from whistress.model.model import WhiStress, WhiStressPhn, WhiStressPhnIa

from utils import StressDataset, MyCollate
from metrics import compute_prf_metrics
from corpora import SUPPORTED_CORPORA, load_corpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_ckpt_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument("--exp_dir", type=str, default="./exp/baseline")
    parser.add_argument("--corpus", choices=SUPPORTED_CORPORA, default="tinystress")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    parser.add_argument("--results_dir", type=Path, default=None)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    pretrained_ckpt_dir = Path(args.pretrained_ckpt_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(pretrained_ckpt_dir / "metadata.json", "r") as fn:
        metadata = json.load(fn)
    
    model = get_loaded_model(device=device, metadata=metadata)

    dataset = load_corpus(args.corpus, args.split, args.data_root / "raw")
    processed_dir = args.data_root / "processed" / args.corpus / args.split
    data_collate = MyCollate(processor=model.processor)
    val_loader = DataLoader(StressDataset(hf_dataset_or_path=dataset, model=model, processed_dir=str(processed_dir)), batch_size=args.batch_size, collate_fn=data_collate)

    best_f1, best_epoch, metrics_log = -1.0, -1, []
    
    # === Validation ===
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation"):
            audio_array = [x["array"] for x in batch["audio_input"]]
            ids = [x for x in batch["id"]]
            stress_pattern_binary = [x for x in batch["stress_pattern_binary"]]
            transcription = [x for x in batch["transcription"]]
            print(ids[0])
            word_labels_batch = batch["labels_head"]
            
            input_features = model.processor.feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")["input_features"].to(device)
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            labels = batch["labels_head"].to(device)
            phone_ids = batch["phone_ids"].to(device)
            phone_labels_head = batch["phone_labels_head"].to(device)
            token_pos_ids = batch["token_pos_ids"].to(device)
            word_ids = batch["word_ids"].to(device)

            output = model(
                input_features=input_features, 
                decoder_input_ids=decoder_input_ids, 
                labels_head=labels, 
                phone_ids=phone_ids, 
                phone_labels_head=phone_labels_head,
                token_pos_ids=token_pos_ids,
                word_ids=word_ids)
            
            preds = output.preds.view(-1).tolist()
            labels_flat = labels.view(-1).tolist()
            
            for p, l in zip(preds, labels_flat):
                if l != -100:
                    all_preds.append(p)
                    all_labels.append(l)

    prf = compute_prf_metrics(all_preds, all_labels)
    print(f"Precision: {prf['precision']:.4f}, Recall: {prf['recall']:.4f}, F1: {prf['f1']:.4f}")
    results_dir = args.results_dir or exp_dir / "test" / args.corpus
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "dataset": args.corpus,
        "split": args.split,
        "evaluation_stage": 2,
        "evaluation_mode": "teacher_forced",
        "metric_level": "whisper_token",
        "num_samples": len(dataset),
        "num_labeled_tokens": len(all_labels),
        **prf,
    }
    with (results_dir / "stage2_metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)
