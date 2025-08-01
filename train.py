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

from utils import StressDataset, MyCollate
from metrics import compute_prf_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper_tag", type=str, default="openai/whisper-small.en")
    parser.add_argument("--pretrained_ckpt_dir", type=str)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--init_lr", type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--layer_for_head', type=int, default=9)
    parser.add_argument("--patience", type=int, default=-1)
    parser.add_argument("--accumulate_gradient_steps", type=int, default=1)
    parser.add_argument("--exp_dir", type=str, default="./exp/baseline")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    init_lr = args.init_lr
    exp_dir = args.exp_dir
    if args.pretrained_ckpt_dir:
        pretrained_ckpt_dir = Path(args.pretrained_ckpt_dir)
    patience = args.patience if args.patience != -1 else args.epochs
    
    whisper_tag = args.whisper_tag
    wandb.init(project="whistress", name=args.exp_dir, config=vars(args), mode="online")

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
    raw_train_dataset = dataset["train"].train_test_split(test_size=0.1, seed=seed)
    dataset["train"] = raw_train_dataset["train"]
    dataset["val"] = raw_train_dataset["test"]
    
    data_collate = MyCollate(processor=model.processor)
    train_loader = DataLoader(StressDataset(dataset["train"], model), batch_size=args.batch_size, shuffle=True, collate_fn=data_collate)
    val_loader = DataLoader(StressDataset(dataset["val"], model), batch_size=args.batch_size, collate_fn=data_collate)

    best_f1, best_epoch, metrics_log = -1.0, -1, []
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        train_all_preds, train_all_labels = [], []
        for step, batch in enumerate(tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training")):
            audio_array = [x["array"] for x in batch["audio_input"]]

            input_features = model.processor.feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")["input_features"].to(device)
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            labels = batch["labels_head"].to(device)

            output = model(input_features=input_features, decoder_input_ids=decoder_input_ids, labels_head=labels)
            loss = output.loss / args.accumulate_gradient_steps
            loss.backward()
            
            if (step + 1) % args.accumulate_gradient_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # collect predictions for PRF
            preds = output.preds.view(-1).tolist()
            labels_flat = labels.view(-1).tolist()
            for p, l in zip(preds, labels_flat):
                if l != -100:
                    train_all_preds.append(p)
                    train_all_labels.append(l)
            
            total_loss += loss.item()

        train_prf = compute_prf_metrics(train_all_preds, train_all_labels)
        print(f"[Epoch {epoch+1}] - Train Loss: {total_loss / len(train_loader):.4f}, Precision: {train_prf['precision']:.4f}, Recall: {train_prf['recall']:.4f}, F1: {train_prf['f1']:.4f}")

        # === Validation ===
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Epoch {epoch+1}] Validation"):
                audio_array = [x["array"] for x in batch["audio_input"]]

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
            
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                break
        
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": total_loss / len(train_loader),
            "train/precision": train_prf["precision"],
            "train/recall": train_prf["recall"],
            "train/f1": train_prf["f1"],
            "val/precision": prf["precision"],
            "val/recall": prf["recall"],
            "val/f1": prf["f1"]
        })

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=4)

    print(f"\n🏆 Final best model at epoch {best_epoch} with F1 = {best_f1:.4f}")
