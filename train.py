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
from whistress.model.model import (
    WhiStress,
    WhiStressPos,
    WhiStressPhn,
    WhiStressPhnPos,
    WhiStressPhnIa,
)

from utils import StressDataset, MyCollate, load_from_json, save_to_json
from metrics import compute_prf_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_conf", type=str, default="conf/baseline.json")
    parser.add_argument("--pretrained_ckpt_dir", type=str)
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument("--exp_dir", type=str, default="./exp/baseline")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    training_config = load_from_json(args.train_conf)
    train_args, model_args = training_config[0], training_config[1]

    exp_dir = args.exp_dir
    if args.pretrained_ckpt_dir:
        pretrained_ckpt_dir = Path(args.pretrained_ckpt_dir)
     
    epochs = train_args["epochs"]
    init_lr = train_args["init_lr"]
    patience = train_args["patience"] if train_args["patience"] != -1 else epochs
    batch_size = train_args["batch_size"]
    accumulate_gradient_steps = train_args["accumulate_gradient_steps"]

    model_type = model_args["model_type"]
    whisper_tag = model_args["whisper_tag"]
    loss_lambdas = model_args["loss_lambdas"]
    layer_for_head = model_args["layer_for_head"]
    pos_bias_config = model_args.get("pos_bias_config", None)
    initialization_config = model_args.get("initialization_config", {})
    #wandb.init(project="whistress", name=args.exp_dir, config=vars(args), mode="online")

    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    best_ckpt_dir = os.path.join(exp_dir, "best")
    exp_dir = Path(exp_dir)
    ckpt_dir = Path(ckpt_dir)
    best_ckpt_dir = Path(best_ckpt_dir)

    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    train_conf_path = os.path.join(args.exp_dir, 'train_conf.json')
    save_to_json(train_args, train_conf_path)
    model_conf_path = os.path.join(args.exp_dir, 'model_conf.json')
    save_to_json(model_args, model_conf_path)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    hyper_params = {
        "model_type": model_type,
        "layer_for_head": layer_for_head,
        "whisper_tag": whisper_tag
    }
    if model_type in ["WhiStressPos", "WhiStressPhnPos"]:
        hyper_params["pos_bias_config"] = pos_bias_config
        hyper_params["initialization_config"] = initialization_config

    is_pos_model = model_type in ["WhiStressPos", "WhiStressPhnPos"]
    train_from_scratch = initialization_config.get("train_from_scratch", True)
    parent_checkpoint_dir = initialization_config.get("checkpoint_dir")
    freeze_pretrained_heads = initialization_config.get(
        "freeze_pretrained_heads", False
    )
    # A resume restores the complete POS experiment and intentionally ignores
    # parent-initialization settings recorded for experiment provenance.
    if is_pos_model and not args.resume:
        if train_from_scratch:
            if parent_checkpoint_dir is not None:
                raise ValueError(
                    "checkpoint_dir must be null when train_from_scratch=True"
                )
            if freeze_pretrained_heads:
                raise ValueError(
                    "freeze_pretrained_heads must be false when "
                    "train_from_scratch=True"
                )
        elif parent_checkpoint_dir is None:
            raise ValueError(
                "checkpoint_dir is required when train_from_scratch=False"
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = WhisperConfig.from_pretrained(whisper_tag)

    if model_type == "WhiStress":
        print("Train WhiStress")
        model = WhiStress(config=config, 
                    layer_for_head=layer_for_head, 
                    whisper_backbone_name=whisper_tag,
                    loss_lambdas=loss_lambdas).to(device)
    elif model_type == "WhiStressPos":
        print("Train WhiStressPos")
        model = WhiStressPos(config=config,
                    layer_for_head=layer_for_head,
                    whisper_backbone_name=whisper_tag,
                    loss_lambdas=loss_lambdas,
                    pos_bias_config=pos_bias_config,
                    freeze_pretrained_heads=freeze_pretrained_heads).to(device)
    elif model_type == "WhiStressPhn":
        print("Train WhiStressPhn")
        model = WhiStressPhn(config=config, 
                    layer_for_head=layer_for_head, 
                    whisper_backbone_name=whisper_tag, 
                    num_phones=39,
                    loss_lambdas=loss_lambdas).to(device)
    elif model_type == "WhiStressPhnIa":
        print("Train WhiStressPhnIa")
        model = WhiStressPhnIa(config=config, 
                    layer_for_head=layer_for_head, 
                    whisper_backbone_name=whisper_tag, 
                    num_phones=39,
                    loss_lambdas=loss_lambdas).to(device)
    elif model_type == "WhiStressPhnPos":
        print("Train WhiStressPhnPos")
        model = WhiStressPhnPos(config=config,
                    layer_for_head=layer_for_head,
                    whisper_backbone_name=whisper_tag,
                    num_phones=39,
                    loss_lambdas=loss_lambdas,
                    pos_bias_config=pos_bias_config,
                    freeze_pretrained_heads=freeze_pretrained_heads).to(device)
    else:
        raise ValueError(f"model_type {model_type} hasn't been implemented yet.")

    model.processor.tokenizer.model_input_names = [
        "input_ids",
        "attention_mask",
        "labels_head",
    ]

    if args.resume:
        if not args.pretrained_ckpt_dir:
            raise ValueError("--pretrained_ckpt_dir is required with --resume")
        resume_model_path = pretrained_ckpt_dir / "model.pt"
        if not resume_model_path.exists():
            raise ValueError(f"Resume checkpoint not found: {resume_model_path}")
        model.load_state_dict(torch.load(resume_model_path, map_location=device))
    elif is_pos_model and not train_from_scratch:
        parent_checkpoint_dir = Path(parent_checkpoint_dir)
        metadata_path = parent_checkpoint_dir / "metadata.json"
        model_path = parent_checkpoint_dir / "model.pt"
        if not metadata_path.exists() or not model_path.exists():
            raise ValueError(
                f"checkpoint_dir must contain model.pt and metadata.json: "
                f"{parent_checkpoint_dir}"
            )
        with open(metadata_path, "r") as fn:
            parent_metadata = json.load(fn)
        expected_parent_type = {
            "WhiStressPos": "WhiStress",
            "WhiStressPhnPos": "WhiStressPhn",
        }[model_type]
        if parent_metadata.get("model_type") != expected_parent_type:
            raise ValueError(
                f"{model_type} requires a {expected_parent_type} checkpoint, "
                f"got {parent_metadata.get('model_type')}"
            )
        state_dict = torch.load(model_path, map_location=device)
        load_result = model.load_state_dict(state_dict, strict=False)
        invalid_missing_keys = [
            key for key in load_result.missing_keys
            if not key.startswith("pos_bias.")
        ]
        if invalid_missing_keys or load_result.unexpected_keys:
            raise ValueError(
                "Incompatible parent checkpoint: "
                f"missing_keys={load_result.missing_keys}, "
                f"unexpected_keys={load_result.unexpected_keys}"
            )

    # Enforce the configured freeze policy before selecting optimizer parameters.
    model.train()
    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad], lr=init_lr
    )

    dataset = load_dataset("slprl/TinyStress-15K")
    raw_train_dataset = dataset["train"].train_test_split(test_size=0.1, seed=seed)
    dataset["train"] = raw_train_dataset["train"]
    dataset["val"] = raw_train_dataset["test"]
    
    data_collate = MyCollate(processor=model.processor)
    train_loader = DataLoader(StressDataset(hf_dataset_or_path=dataset["train"], model=model, processed_dir="data/train"), batch_size=batch_size, shuffle=True, collate_fn=data_collate)
    val_loader = DataLoader(StressDataset(hf_dataset_or_path=dataset["val"], model=model, processed_dir="data/valid"), batch_size=batch_size, collate_fn=data_collate)

    best_f1, best_epoch, metrics_log = -1.0, -1, []
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss, total_loss_main, total_loss_wsd, total_loss_wsl = 0.0, 0.0, 0.0, 0.0
        train_all_preds, train_all_labels = [], []
        for step, batch in enumerate(tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training")):
            audio_array = [x["array"] for x in batch["audio_input"]]

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
                        word_ids=word_ids
                    )
            loss_main = output.loss_main
            loss_wsd = output.loss_wsd
            loss_wsl = output.loss_wsl
            loss = output.loss
            
            loss = loss / accumulate_gradient_steps
            loss.backward()
            
            if (step + 1) % accumulate_gradient_steps == 0:
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
            total_loss_main += loss_main.item()
            if loss_wsd is not None:
                total_loss_wsd += loss_wsd.item()
            if loss_wsl is not None:
                total_loss_wsl += loss_wsl.item()

        train_prf = compute_prf_metrics(train_all_preds, train_all_labels)
        print(f"[Epoch {epoch+1}] - Train Loss: {total_loss / len(train_loader):.4f}, Main Loss: {total_loss_main / len(train_loader):.4f}, Phn Loss: {total_loss_wsd / len(train_loader):.4f}, WSL: {total_loss_wsl / len(train_loader):.4f}, Precision: {train_prf['precision']:.4f}, Recall: {train_prf['recall']:.4f}, F1: {train_prf['f1']:.4f}")

        # === Validation ===
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Epoch {epoch+1}] Validation"):
                audio_array = [x["array"] for x in batch["audio_input"]]

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
                        word_ids=word_ids
                    )

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
        
        #wandb.log({
        #    "epoch": epoch + 1,
        #    "train/loss": total_loss / len(train_loader),
        #    "train/precision": train_prf["precision"],
        #    "train/recall": train_prf["recall"],
        #    "train/f1": train_prf["f1"],
        #    "val/precision": prf["precision"],
        #    "val/recall": prf["recall"],
        #    "val/f1": prf["f1"]
        #})

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=4)

    print(f"\n🏆 Final best model at epoch {best_epoch} with F1 = {best_f1:.4f}")
