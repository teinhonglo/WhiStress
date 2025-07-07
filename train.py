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

def preprocess(example, model):
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
    word_idx = 0
    for token in decoded_tokens:
        if token in model.processor.tokenizer.all_special_tokens:
            token_labels.append(-100)
            continue

        if token.startswith(" "):  # new word
            label = binary[word_idx] if word_idx < len(binary) else 0
            token_labels.append(label)
            word_idx += 1
        else:  # subword
            label = binary[word_idx - 1] if word_idx - 1 < len(binary) else 0
            token_labels.append(label)

    # 5. LEFT SHIFT → align label with logits[t] = prediction of token[t+1]
    token_labels = token_labels[1:] + [-100]

    # 6. store results
    example["decoder_input_ids"] = input_ids
    example["labels_head"] = token_labels
    return example

class StressDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

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
            "id": item["id"]
        }

@dataclass
class MyCollate:
    processor: WhisperProcessor

    def __call__(self, batch):
        decoder_input_ids = [torch.tensor(b["decoder_input_ids"]) for b in batch]
        labels_head = [torch.tensor(b["labels_head"]) for b in batch]
        
        return {
            "audio": [b["audio"] for b in batch],
            "audio_input": [b["audio_input"] for b in batch],
            "decoder_input_ids": pad_sequence(decoder_input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id),
            "labels_head": pad_sequence(labels_head, batch_first=True, padding_value=-100),
            "transcription": [b["transcription"] for b in batch],
            "id": [b["id"] for b in batch]
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
    if args.pretrained_ckpt_dir:
        pretrained_ckpt_dir = Path(args.pretrained_ckpt_dir)
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

    config = WhisperConfig.from_pretrained(whisper_tag)
    model = WhiStress(config=config, layer_for_head=args.layer_for_head, whisper_backbone_name=whisper_tag).to(device)
    model.processor.tokenizer.model_input_names = [
        "input_ids",
        "attention_mask",
        "labels_head",
    ]

    if args.resume and (pretrained_ckpt_dir / "model.pt").exists():
        model.load_state_dict(torch.load(pretrained_ckpt_dir / "model.pt"))

    optimizer = AdamW(model.parameters(), lr=init_lr)

    dataset = load_dataset("slprl/TinyStress-15K")
    dataset["train"] = dataset["train"].map(lambda x: preprocess(x, model=model), num_proc=4)
    dataset["test"] = dataset["test"].map(lambda x: preprocess(x, model=model), num_proc=4)

    data_collate = MyCollate(processor=model.processor)
    train_loader = DataLoader(StressDataset(dataset["train"]), batch_size=args.batch_size, shuffle=True, collate_fn=data_collate)
    val_loader = DataLoader(StressDataset(dataset["test"]), batch_size=args.batch_size, collate_fn=data_collate)

    best_f1, best_epoch, metrics_log = -1.0, -1, []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training"):
            audio_array = [x["array"] for x in batch["audio_input"]]
            word_labels_batch = batch["labels_head"]

            input_features = model.processor.feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")["input_features"].to(device)
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            labels = batch["labels_head"].to(device)

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
                audio_array = [x["array"] for x in batch["audio_input"]]
                word_labels_batch = batch["labels_head"]

                input_features = model.processor.feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")["input_features"].to(device)
                decoder_input_ids = batch["decoder_input_ids"].to(device)
                labels = batch["labels_head"].to(device)

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
