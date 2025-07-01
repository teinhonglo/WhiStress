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

def preprocess(example, tokenizer=None):
    binary = compute_stress_binary(example["transcription"], example["emphasis_indices"])
    example["stress_pattern"] = {"binary": binary}
    array = prepare_audio(audio=example["audio"])
    example["audio_input"] = {
        "array": array,
        "sampling_rate": 16000
    }

    words = example["transcription"].strip().split()
    tokenized = tokenizer(
        words,
        return_tensors=None,
        padding=False,            
        truncation=False,         
        is_split_into_words=True,
    )
    word_ids = tokenized.word_ids()
    token_labels = [
        binary[word_id] if word_id is not None and word_id < len(binary) else -100
        for word_id in word_ids
    ]
    decoder_input_ids = tokenized["input_ids"]
    '''
    decoder_input_ids [50257, 50362, 1375, 1392, 845, 6568, 290, 2067, 284, 27318, 656, 262, 7604, 13, 50256]
    word_ids [None, None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, None]
    token_labels [-100, -100, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100]
    '''
    example["decoder_input_ids"] = decoder_input_ids
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
    tokenizer: WhisperTokenizerFast

    def __call__(self, batch):
        decoder_input_ids = [torch.tensor(b["decoder_input_ids"]) for b in batch]
        labels_head = [torch.tensor(b["labels_head"]) for b in batch]
        
        return {
            "audio": [b["audio"] for b in batch],
            "audio_input": [b["audio_input"] for b in batch],
            "decoder_input_ids": pad_sequence(decoder_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id),
            "labels_head": pad_sequence(labels_head, batch_first=True, padding_value=-100),
            "transcription": [b["transcription"] for b in batch],
            "id": [b["id"] for b in batch]
        }

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
    exp_dir = args.exp_dir
    pretrained_ckpt_dir = Path(args.pretrained_ckpt_dir)
    whisper_tag = args.whisper_tag

    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    best_ckpt_dir = os.path.join(exp_dir, "best")
    exp_dir = Path(exp_dir)
    ckpt_dir = Path(ckpt_dir)
    best_ckpt_dir = Path(best_ckpt_dir)

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

    tokenizer = WhisperTokenizerFast.from_pretrained(whisper_tag, add_prefix_space=True)
    processor = WhisperProcessor.from_pretrained(whisper_tag)
    processor.tokenizer = tokenizer
    config = WhisperConfig.from_pretrained(whisper_tag)
    model = WhiStress(config=config, layer_for_head=args.layer_for_head).to(device)

    #model.load_state_dict(torch.load(pretrained_ckpt_dir / "model.pt"))
    with open(pretrained_ckpt_dir / "metadata.json", "r") as fn:
        metadata = json.load(fn)
    
    model = get_loaded_model(device="cuda", metadata=metadata)

    dataset = load_dataset("slprl/TinyStress-15K")
    dataset["test"] = dataset["test"].map(lambda x: preprocess(x, tokenizer=tokenizer), num_proc=4)

    data_collate = MyCollate(tokenizer=tokenizer)
    val_loader = DataLoader(StressDataset(dataset["test"]), batch_size=args.batch_size, collate_fn=data_collate)

    best_f1, best_epoch, metrics_log = -1.0, -1, []
    
    # === Validation ===
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation"):
            audio_array = [x["array"] for x in batch["audio_input"]]
            word_labels_batch = batch["labels_head"]
            
            input_features = processor.feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")["input_features"].to(device)
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
    print(f"Precision: {prf['precision']:.4f}, Recall: {prf['recall']:.4f}, F1: {prf['f1']:.4f}")
