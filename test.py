from pathlib import Path
import os
import argparse
import json
import random
import numpy as np
from dataclasses import dataclass, field

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

from utils import StressDataset, MyCollate
from metrics import compute_prf_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper_tag", type=str, default="openai/whisper-small.en")
    parser.add_argument("--pretrained_ckpt_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--init_lr", type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--layer_for_head', type=int, default=9)
    parser.add_argument("--exp_dir", type=str, default="./exp/baseline")
    args = parser.parse_args()

    init_lr = args.init_lr
    exp_dir = Path(args.exp_dir)
    pretrained_ckpt_dir = Path(args.pretrained_ckpt_dir)
    whisper_tag = args.whisper_tag

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = WhisperConfig.from_pretrained(whisper_tag)
    model = WhiStress(config=config, layer_for_head=args.layer_for_head, whisper_backbone_name=whisper_tag).to(device)
    model.processor.tokenizer.model_input_names = [
        "input_ids",
        "attention_mask",
        "labels_head",
    ]

    #model.load_state_dict(torch.load(pretrained_ckpt_dir / "model.pt"))
    with open(pretrained_ckpt_dir / "metadata.json", "r") as fn:
        metadata = json.load(fn)
    
    model = get_loaded_model(device="cuda", metadata=metadata)

    dataset = load_dataset("slprl/TinyStress-15K")
    data_collate = MyCollate(processor=model.processor)
    val_loader = DataLoader(StressDataset(hf_dataset_or_path=dataset["test"], model=model, processed_dir="data/test"), batch_size=args.batch_size, collate_fn=data_collate)

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
            output = model(input_features=input_features, decoder_input_ids=decoder_input_ids, labels_head=labels)
            
            preds = output.preds.view(-1).tolist()
            labels_flat = labels.view(-1).tolist()
            
            #print("prediction", preds)
            #print("ground_truth", labels_flat)
            #print("stress_pattern_binary", stress_pattern_binary[0])
            #print("transcription", transcription[0].split())
            
            for p, l in zip(preds, labels_flat):
                if l != -100:
                    all_preds.append(p)
                    all_labels.append(l)

    prf = compute_prf_metrics(all_preds, all_labels)
    print(f"Precision: {prf['precision']:.4f}, Recall: {prf['recall']:.4f}, F1: {prf['f1']:.4f}")
