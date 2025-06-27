from pathlib import Path
import os
import argparse
import json
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperTokenizerFast, WhisperConfig, AdamW
from tqdm import tqdm
import evaluate
import torch.nn.functional as F

from whistress.inference_client.utils import prepare_audio, save_model_parts
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

def prerpocess(example):
    binary = compute_stress_binary(example["transcription"], example["emphasis_indices"])
    example["stress_pattern"] = {"binary": binary}
    example["audio"]["array"] = prepare_audio(audio=example["audio"], target_sr=16000)
    example["audio"]["sampling_rate"] = 16000
    return example

def collate(batch):
    return {
        "audio": [b["audio"] for b in batch],
        "transcription": [b["transcription"].split() for b in batch],
        "labels_head": [b["labels_head"] for b in batch],
    }

class StressDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "audio": item["audio"],
            "transcription": item["transcription"],
            "labels_head": item["stress_pattern"]["binary"],
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper_tag", type=str, default="openai/whisper-small.en")
    parser.add_argument("--pretrained_ckpt_dir", type=str)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--init_lr", type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--layer_for_head', type=int, default=9)
    parser.add_argument("--exp_dir", type=str, default="./exp/baseline")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    init_lr = args.init_lr
    exp_dir = args.exp_dir
    pretrained_ckpt_dir = args.pretrained_ckpt_dir
    whisper_tag = args.whisper_tag

    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    best_ckpt_dir = os.path.join(exp_dir, "best")
    exp_dir = Path(exp_dir)
    ckpt_dir = Path(ckpt_dir)
    best_ckpt_dir = Path(best_ckpt_dir)

    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

    hyper_params = {"epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "init_lr": args.init_lr,
                    "layer_for_head": args.layer_for_head,
                    "whisper_tag": args.whisper_tag
                    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = WhisperTokenizerFast.from_pretrained(whisper_tag, add_prefix_space=True)
    processor = WhisperProcessor.from_pretrained(whisper_tag)
    processor.tokenizer = tokenizer
    config = WhisperConfig.from_pretrained(whisper_tag)
    model = WhiStress(config=config, layer_for_head=args.layer_for_head).to(device)

    if args.resume and (pretrained_ckpt_dir / "model.pt").exists():
        model.load_state_dict(torch.load(pretrained_ckpt_dir / "model.pt"))

    optimizer = AdamW(model.parameters(), lr=init_lr)

    dataset = load_dataset("slprl/TinyStress-15K")
    dataset["train"] = dataset["train"].map(prerpocess, num_proc=4)
    dataset["test"] = dataset["test"].map(prerpocess, num_proc=4)

    train_loader = DataLoader(StressDataset(dataset["train"]), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(StressDataset(dataset["test"]), batch_size=args.batch_size, collate_fn=collate)

    best_f1, best_epoch, metrics_log = -1.0, -1, []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training"):
            audio_array = [x["array"] for x in batch["audio"]]
            word_labels_batch = batch["labels_head"]

            input_features = processor.feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")["input_features"].to(device)

            tokenized = processor.tokenizer(
                batch["transcription"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=50,
                is_split_into_words=True,
            )
            decoder_input_ids = tokenized["input_ids"].to(device)
            word_ids_batch = [tokenized.word_ids(batch_index=i) for i in range(len(word_labels_batch))]
            
            token_labels = []
            for word_ids, word_labels in zip(word_ids_batch, word_labels_batch):
                labels = [
                    word_labels[word_id] if word_id is not None and word_id < len(word_labels) else -100
                    for word_id in word_ids
                ]
                token_labels.append(torch.tensor(labels))
            labels = torch.nn.utils.rnn.pad_sequence(token_labels, batch_first=True, padding_value=-100).to(device)
            
            #print(f"decoder_input_ids {decoder_input_ids} {decoder_input_ids.shape}")
            #print(f"labels_head {labels} {labels.shape}")

            output = model(input_features=input_features, decoder_input_ids=decoder_input_ids, labels_head=labels)
            loss = output.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Train Loss: {total_loss / len(train_loader):.4f}")

        # === Validation ===
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Epoch {epoch+1}] Validation"):
                audio_array = [x["array"] for x in batch["audio"]]
                word_labels_batch = batch["labels_head"]

                input_features = processor.feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")["input_features"].to(device)

                tokenized = processor.tokenizer(
                    batch["transcription"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=50,
                    is_split_into_words=True,
                )
                decoder_input_ids = tokenized["input_ids"].to(device)
                word_ids_batch = [tokenized.word_ids(batch_index=i) for i in range(len(word_labels_batch))]

                token_labels = []
                for word_ids, word_labels in zip(word_ids_batch, word_labels_batch):
                    labels = [
                        word_labels[word_id] if word_id is not None and word_id < len(word_labels) else -100
                        for word_id in word_ids
                    ]
                    token_labels.append(torch.tensor(labels))
                labels = torch.nn.utils.rnn.pad_sequence(token_labels, batch_first=True, padding_value=-100).to(device)

                output = model(input_features=input_features, decoder_input_ids=decoder_input_ids, labels_head=labels)

                preds = output.preds.view(-1).tolist()
                labels_flat = labels.view(-1).tolist()
                for p, l in zip(preds, labels_flat):
                    if l != -100:
                        all_preds.append(p)
                        all_labels.append(l)

        prf = compute_prf_metrics(all_preds, all_labels)
        print(f"[Epoch {epoch+1}] Precision: {prf['precision']:.4f}, Recall: {prf['recall']:.4f}, F1: {prf['f1']:.4f}")
        metrics_log.append({"epoch": epoch+1, **prf})

        torch.save(model.state_dict(), ckpt_dir / f"epoch{epoch+1}.pt")
        if prf["f1"] > best_f1:
            best_f1, best_epoch = prf["f1"], epoch + 1
            torch.save(model.state_dict(), best_ckpt_dir / "model.pt")
            save_model_parts(model, save_dir=best_ckpt_dir, metadata=hyper_params)
            print(f"✅ Best model updated at epoch {best_epoch} with F1 = {best_f1:.4f}")

            with open(exp_dir / "best.log", "w") as f:
                json.dump(metrics_log[-1], f, indent=4)

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=4)

    print(f"\n🏆 Final best model at epoch {best_epoch} with F1 = {best_f1:.4f}")
