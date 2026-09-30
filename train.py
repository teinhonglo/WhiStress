from pathlib import Path
import os
import argparse
import json
import random
import math
import numpy as np
from dataclasses import dataclass, field
import wandb

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperTokenizerFast, WhisperConfig, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import evaluate
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from whistress.inference_client.utils import prepare_audio, save_model_parts, get_loaded_model
from whistress.model.model import (
    WhiStress,
    ProWhiStress,
    WhiStressPos,
    WhiStressPhn,
    WhiStressPhnPairedResidual,
    WhiStressPhnLocusCoupled,
    WhiStressPhnRelativeLocusCoupled,
    WhiStressPhnStaticPosRelativeLocusCoupled,
    WhiStressPhnLocusCoupledRealization,
    WhiStressPhnIa,
)

from utils import StressDataset, MyCollate, load_from_json, save_to_json
from metrics import compute_prf_metrics

def prepare_model_inputs(model, batch, device):
    audio_array = [x["array"] for x in batch["audio_input"]]
    inputs = {
        "input_features": model.processor.feature_extractor(
            audio_array, sampling_rate=16000, return_tensors="pt"
        )["input_features"].to(device),
    }
    for key in (
        "decoder_input_ids", "labels_head", "phone_ids", "phone_labels_head",
        "token_pos_ids", "word_ids",
    ):
        inputs[key] = batch[key].to(device)
    if getattr(model, "requires_word_alignment", False):
        for key in ("phone_word_ids", "phone_vowel_mask"):
            inputs[key] = batch[key].to(device)
    return inputs


def evaluate_validation(model, val_loader, device):
    model.eval()
    predictions, references = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            inputs = prepare_model_inputs(model, batch, device)
            output = model(**inputs)
            labels = inputs["labels_head"]
            valid = labels.ne(-100)
            predictions.extend(output.preds[valid].tolist())
            references.extend(labels[valid].tolist())
    return compute_prf_metrics(predictions, references)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_conf", type=str, default="conf/baseline.json")
    parser.add_argument("--pretrained_ckpt_dir", type=str)
    parser.add_argument('--seed', type=int, default=None)
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
    seed = args.seed if args.seed is not None else train_args.get("seed", 66)
    split_seed = train_args.get("split_seed", seed)
    validation_ratio = train_args.get("validation_ratio", 0.1)
    eval_steps = train_args.get("eval_steps", 0)
    save_steps = train_args.get("save_steps", 0)
    if accumulate_gradient_steps < 1 or eval_steps < 0 or save_steps < 0:
        raise ValueError("Invalid accumulation, evaluation or checkpoint interval")
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be in (0, 1)")
    train_args["seed"] = seed

    model_type = model_args["model_type"]
    whisper_tag = model_args["whisper_tag"]
    loss_lambdas = model_args["loss_lambdas"]
    layer_for_head = model_args["layer_for_head"]
    pos_bias_config = model_args.get("pos_bias_config", None)
    paired_residual_config = model_args.get("paired_residual_config", None)
    locus_coupling_config = model_args.get("locus_coupling_config", None)
    relation_loss_config = model_args.get("relation_loss_config", None)
    mil_loss_config = model_args.get("mil_loss_config", None)
    initialization_config = model_args.get("initialization_config", {})
    prowhistress_config = model_args.get("prowhistress_config", None)
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
    hyper_params.update({
        "seed": seed, "split_seed": split_seed,
        "validation_ratio": validation_ratio,
    })
    if model_type in [
        "WhiStressPos",
        "WhiStressPhnStaticPosRelativeLocusCoupled",
    ]:
        hyper_params["pos_bias_config"] = pos_bias_config
    if model_type == "WhiStressPos":
        hyper_params["initialization_config"] = initialization_config
    if model_type == "WhiStressPhnPairedResidual":
        hyper_params["paired_residual_config"] = paired_residual_config or {}
        hyper_params["relation_loss_config"] = relation_loss_config or {}
        hyper_params["mil_loss_config"] = mil_loss_config or {}
    if model_type in [
        "WhiStressPhnLocusCoupled",
        "WhiStressPhnRelativeLocusCoupled",
        "WhiStressPhnStaticPosRelativeLocusCoupled",
        "WhiStressPhnLocusCoupledRealization",
    ]:
        hyper_params["locus_coupling_config"] = locus_coupling_config or {}

    is_pos_model = model_type == "WhiStressPos"
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
    elif model_type == "ProWhiStress":
        model = ProWhiStress(
            config=config,
            layer_for_head=layer_for_head,
            whisper_backbone_name=whisper_tag,
            loss_lambdas=loss_lambdas,
            prowhistress_config=prowhistress_config,
        ).to(device)
        hyper_params["prowhistress_config"] = model.prowhistress_config
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
    elif model_type == "WhiStressPhnPairedResidual":
        print("Train WhiStressPhnPairedResidual")
        model = WhiStressPhnPairedResidual(
                    config=config,
                    layer_for_head=layer_for_head,
                    whisper_backbone_name=whisper_tag,
                    num_phones=39,
                    loss_lambdas=loss_lambdas,
                    paired_residual_config=paired_residual_config,
                    relation_loss_config=relation_loss_config,
                    mil_loss_config=mil_loss_config).to(device)
    elif model_type == "WhiStressPhnLocusCoupled":
        print("Train WhiStressPhnLocusCoupled")
        model = WhiStressPhnLocusCoupled(
                    config=config,
                    layer_for_head=layer_for_head,
                    whisper_backbone_name=whisper_tag,
                    num_phones=39,
                    loss_lambdas=loss_lambdas,
                    locus_coupling_config=locus_coupling_config).to(device)
    elif model_type == "WhiStressPhnLocusCoupledRealization":
        print("Train WhiStressPhnLocusCoupledRealization")
        model = WhiStressPhnLocusCoupledRealization(
                    config=config,
                    layer_for_head=layer_for_head,
                    whisper_backbone_name=whisper_tag,
                    num_phones=39,
                    loss_lambdas=loss_lambdas,
                    locus_coupling_config=locus_coupling_config).to(device)
    elif model_type == "WhiStressPhnRelativeLocusCoupled":
        print("Train WhiStressPhnRelativeLocusCoupled")
        model = WhiStressPhnRelativeLocusCoupled(
                    config=config,
                    layer_for_head=layer_for_head,
                    whisper_backbone_name=whisper_tag,
                    num_phones=39,
                    loss_lambdas=loss_lambdas,
                    locus_coupling_config=locus_coupling_config).to(device)
    elif model_type == "WhiStressPhnStaticPosRelativeLocusCoupled":
        print("Train WhiStressPhnStaticPosRelativeLocusCoupled")
        model = WhiStressPhnStaticPosRelativeLocusCoupled(
                    config=config,
                    layer_for_head=layer_for_head,
                    whisper_backbone_name=whisper_tag,
                    num_phones=39,
                    loss_lambdas=loss_lambdas,
                    locus_coupling_config=locus_coupling_config,
                    pos_bias_config=pos_bias_config).to(device)
    elif model_type == "WhiStressPhnIa":
        print("Train WhiStressPhnIa")
        model = WhiStressPhnIa(config=config, 
                    layer_for_head=layer_for_head, 
                    whisper_backbone_name=whisper_tag, 
                    num_phones=39,
                    loss_lambdas=loss_lambdas).to(device)
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
        expected_parent_type = "WhiStress"
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
    if train_args.get("optimizer") == "adamw_torch":
        # Match Trainer's AdamW: no decay on bias or LayerNorm parameters.
        no_decay_ids = {
            id(param)
            for module in model.modules() if isinstance(module, torch.nn.LayerNorm)
            for param in module.parameters()
        }
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Includes MultiheadAttention.in_proj_bias, matching Trainer.
                target = no_decay if "bias" in name or id(param) in no_decay_ids else decay
                target.append(param)
        optimizer = torch.optim.AdamW([
            {"params": decay, "weight_decay": train_args.get("weight_decay", 0.01)},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=init_lr, betas=(0.9, 0.999), eps=1e-8)
    else:
        optimizer = AdamW(
            [param for param in model.parameters() if param.requires_grad], lr=init_lr
        )

    dataset = load_dataset("slprl/TinyStress-15K")
    raw_train_dataset = dataset["train"].train_test_split(
        test_size=validation_ratio, seed=split_seed,
    )
    dataset["train"] = raw_train_dataset["train"]
    dataset["val"] = raw_train_dataset["test"]
    
    train_processed_dir, val_processed_dir = "data/train", "data/valid"
    if model_type == "ProWhiStress":
        # Fixed split seed 42 across model seeds 42--46, and no reuse of the
        # existing 10%-validation cache for the author's 2%-validation split.
        cache_root = Path("data/processed/tinystress")
        train_processed_dir = str(cache_root / f"train_{dataset['train']._fingerprint}")
        val_processed_dir = str(cache_root / f"valid_{dataset['val']._fingerprint}")
    data_collate = MyCollate(processor=model.processor)
    train_loader = DataLoader(
        StressDataset(
            hf_dataset_or_path=dataset["train"],
            model=model,
            processed_dir=train_processed_dir,
        ),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collate,
    )
    val_loader = DataLoader(
        StressDataset(
            hf_dataset_or_path=dataset["val"],
            model=model,
            processed_dir=val_processed_dir,
        ),
        batch_size=train_args.get("eval_batch_size", batch_size),
        collate_fn=data_collate,
    )

    if not len(train_loader) or not len(val_loader):
        raise ValueError("Training and validation loaders must both be non-empty")
    scheduler = None
    if train_args.get("lr_scheduler") == "linear":
        total_steps = math.ceil(len(train_loader) / accumulate_gradient_steps) * epochs
        warmup_steps = math.ceil(train_args.get("warmup_ratio", 0.0) * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1, best_epoch, metrics_log = -1.0, -1, []
    patience_counter = 0
    global_step, last_eval_step = 0, -1

    def validate(epoch):
        nonlocal best_f1, best_epoch, patience_counter, last_eval_step
        prf = evaluate_validation(model, val_loader, device)
        last_eval_step = global_step
        print(f"[Epoch {epoch:.3f}, Step {global_step}] Validation: {prf}")
        metrics_log.append({"epoch": epoch, "step": global_step, **prf})
        if prf["f1"] > best_f1:
            best_f1, best_epoch = prf["f1"], epoch
            hyper_params["global_step"] = global_step
            torch.save(model.state_dict(), best_ckpt_dir / "model.pt")
            save_model_parts(model, save_dir=best_ckpt_dir, metadata=hyper_params)
            with open(exp_dir / "best.log", "w") as file:
                json.dump(metrics_log[-1], file, indent=4)
            patience_counter = 0
        else:
            patience_counter += 1
        model.train()
        return train_args["patience"] != -1 and patience_counter >= patience

    def save_step_checkpoint():
        torch.save(model.state_dict(), ckpt_dir / f"step{global_step}.pt")
        limit = train_args.get("save_total_limit")
        if limit is not None:
            if limit < 1:
                raise ValueError("save_total_limit must be positive")
            checkpoints = sorted(
                ckpt_dir.glob("step[0-9]*.pt"),
                key=lambda path: int(path.stem[4:]),
            )
            for checkpoint in checkpoints[:-limit]:
                checkpoint.unlink()

    optimizer.zero_grad()
    should_stop = False

    for epoch in range(epochs):
        model.train()
        total_loss, total_loss_main, total_loss_wsd, total_loss_wsl = 0.0, 0.0, 0.0, 0.0
        total_loss_rank, total_loss_mil = 0.0, 0.0
        train_all_preds, train_all_labels = [], []
        for step, batch in enumerate(tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training")):
            model_inputs = prepare_model_inputs(model, batch, device)
            labels = model_inputs["labels_head"]
            output = model(**model_inputs)
            loss_main = output.loss_main
            loss_wsd = output.loss_wsd
            loss_wsl = output.loss_wsl
            loss_rank = output.loss_rank
            loss_mil = output.loss_mil
            loss = output.loss
            
            # Normalize and flush the final partial accumulation group too.
            group_size = min(
                accumulate_gradient_steps,
                len(train_loader) - (step // accumulate_gradient_steps) * accumulate_gradient_steps,
            )
            loss = loss / group_size
            loss.backward()
            
            if (step + 1) % accumulate_gradient_steps == 0 or step + 1 == len(train_loader):
                if train_args.get("max_grad_norm") is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        train_args["max_grad_norm"],
                    )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if eval_steps and global_step % eval_steps == 0:
                    should_stop = validate(epoch + (step + 1) / len(train_loader))
                if save_steps and global_step % save_steps == 0:
                    save_step_checkpoint()
            
            # collect predictions for PRF
            preds = output.preds.view(-1).tolist()
            labels_flat = labels.view(-1).tolist()
            for p, l in zip(preds, labels_flat):
                if l != -100:
                    train_all_preds.append(p)
                    train_all_labels.append(l)
            
            total_loss += output.loss.item()
            total_loss_main += loss_main.item()
            if loss_wsd is not None:
                total_loss_wsd += loss_wsd.item()
            if loss_wsl is not None:
                total_loss_wsl += loss_wsl.item()
            if loss_rank is not None:
                total_loss_rank += loss_rank.item()
            if loss_mil is not None:
                total_loss_mil += loss_mil.item()
            if should_stop:
                break

        train_prf = compute_prf_metrics(train_all_preds, train_all_labels)
        print(
            f"[Epoch {epoch+1}] - Train Loss: {total_loss / len(train_loader):.4f}, "
            f"Main Loss: {total_loss_main / len(train_loader):.4f}, "
            f"Phn Loss: {total_loss_wsd / len(train_loader):.4f}, "
            f"WSL: {total_loss_wsl / len(train_loader):.4f}, "
            f"Rank: {total_loss_rank / len(train_loader):.4f}, "
            f"MIL: {total_loss_mil / len(train_loader):.4f}, "
            f"Precision: {train_prf['precision']:.4f}, "
            f"Recall: {train_prf['recall']:.4f}, F1: {train_prf['f1']:.4f}"
        )

        # Legacy configs validate/save once per epoch. Cover a short step-based
        # run's final step as well if it is not a multiple of eval_steps.
        if not eval_steps or (epoch == epochs - 1 and last_eval_step != global_step):
            should_stop = validate(epoch + 1) or should_stop
        if not save_steps:
            torch.save(model.state_dict(), ckpt_dir / f"epoch{epoch+1}.pt")
        if should_stop:
            print(f"Early stopping at epoch {epoch+1}, step {global_step}")
            break

    if save_steps and global_step % save_steps:
        save_step_checkpoint()

    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=4)

    print(f"\n🏆 Final best model at epoch {best_epoch} with F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()
