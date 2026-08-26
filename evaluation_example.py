import torch
import evaluate
from tqdm import tqdm
from pathlib import Path
from whistress import WhiStressInferenceClient
import pprint
import pyphen
from utils import StressDataset, MyCollate
from torch.utils.data import DataLoader
import argparse
import json

from corpora import SUPPORTED_CORPORA, compute_stress_binary, load_corpus

dic = pyphen.Pyphen(lang='en')

def count_syllables(word):
    hyphenated = dic.inserted(word)
    return len(hyphenated.split('-')) if hyphenated else 1

CURRENT_DIR = Path(__file__).parent

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


def calculate_metrics_on_dataset(dataset, whistress_client, with_transcription=True, device="cpu", skip=False):
    """
    Sample structure example for slp-rl/StressTest dataset:
    # {'id': '98dd4a46-6b59-4817-befc-e35d548465c7',
    #  'transcription': 'You chose to do this?',
    #  'description': 'You voluntarily do this without being forced?',
    #  'intonation': 'You *chose* to do this?',
    #  'interpretation_id': '8dbe4033-2451-4674-bf9d-b8e05c61e9c4',
    #  'audio': {'path': None,
    #   'array': array([-1.22070312e-04, -9.15527344e-05, -6.10351562e-05, ...,
    #           2.44140625e-04,  2.13623047e-04,  2.44140625e-04]),
    #   'sampling_rate': 16000},
    #  'metadata': {'gender': 'male',
    #   'language_code': 'en',
    #   'sample_rate_hertz': 16000,
    #   'voice_name': 'actor'},
    #  'possible_answers': ['Why did you choose this option out of all of them?',
    #   'You voluntarily do this without being forced?'],
    #  'label': 1,
    #  'stress_pattern': {'binary': [0, 1, 0, 0, 0],
    #   'indices': [1],
    #   'words': ['chose']}
    """
    predictions = []
    references = []
    predictions_psd = []
    references_psd = []
    error_cases = []
    num_skipped = 0

    for sample in tqdm(dataset):
        #gt_stresses = sample['stress_pattern']['binary']
        gt_stresses = sample['stress_pattern_binary']
        
        if with_transcription:
            phone_ids = sample['phone_ids'].reshape(1,-1).to(device)
            token_pos_ids = sample['token_pos_ids'].reshape(1,-1).to(device)
            # Transcription
            scored, phone_stress_preds = whistress_client.predict(
                audio=sample['audio'],
                # Using ground truth transcription for evaluating stress prediction ability. 
                # set transcription to None if not available
                transcription=sample['transcription'], 
                return_pairs=True,
                phone_ids=phone_ids,
                token_pos_ids=token_pos_ids
            )
        else:
            scored, phone_stress_preds = whistress_client.predict(
                audio=sample['audio'],
                transcription=None, 
                return_pairs=True
            )
        _, pred_stresses = zip(*scored)
        # Ensure the lengths are the same 
        # When transcription is not provided, predictions should be aligned with the ground truth
        
        #assert len(pred_stresses) == len(gt_stresses), "Length mismatch"
        if len(pred_stresses) != len(gt_stresses):
            print("Length mismatch")
            print(sample['transcription'])
            print(scored)
            print(pred_stresses, len(pred_stresses))
            print(gt_stresses, len(gt_stresses))    
            num_skipped += 1
            continue
        
        words = sample["transcription"].strip().split()
        duration_sec = len(sample['audio']['array']) / sample['audio']['sampling_rate']
        utt_len = len(words)
        speaking_rate = utt_len / duration_sec

        word_results = []
        for i, (gt, pred) in enumerate(zip(gt_stresses, pred_stresses)):
            if gt == 1 and pred == 1:
                tag = "TP"
            elif gt == 0 and pred == 0:
                tag = "TN"
            elif gt == 1 and pred == 0:
                tag = "FN"
            elif gt == 0 and pred == 1:
                tag = "FP"
            else:
                tag = "?"

            word_results.append({
                "index": i,
                "word": words[i],
                "gt": gt,
                "pred": pred,
                "type": tag,
                "word_len": len(words[i]),
                "syllable_count": count_syllables(words[i]),
            })
        
        
        if phone_stress_preds is not None:
            # shape (1, num_phones)
            phone_stress_preds = phone_stress_preds.to("cpu").detach().numpy()[0]
            # shape (num_phones)
            phone_labels_head = sample['phone_labels_head'].to("cpu").detach().numpy()
            predictions_psd.extend(list(phone_stress_preds))
            references_psd.extend(list(phone_labels_head))
        
        error_cases.append({
            "id": str(sample["id"]),
            "source_dataset": sample.get("source_dataset", "unknown"),
            "transcription": sample["transcription"],
            "utt_len": utt_len,
            "utt_duration": duration_sec,
            "speaking_rate": speaking_rate,
            "gt_stresses": gt_stresses,
            "pred_stresses": list(pred_stresses),
            "words": word_results
        })
        
        predictions.extend(pred_stresses)
        references.extend(gt_stresses)

    metrics = compute_prf_metrics(predictions, references, average="binary") if references else None
    metrics_psd = compute_prf_metrics(predictions_psd, references_psd, average="binary") if references_psd else None
    num_samples = len(dataset)
    coverage = {
        "num_samples": num_samples,
        "num_evaluated": num_samples - num_skipped,
        "num_skipped": num_skipped,
        "coverage_rate": (num_samples - num_skipped) / num_samples if num_samples else 0.0,
        "skip_reasons": {"word_length_mismatch": num_skipped},
    }
    return metrics, metrics_psd, error_cases, coverage

def add_stress_pattern(example):
    binary = compute_stress_binary(example["transcription"], example["emphasis_indices"])
    example["stress_pattern"] = {"binary": binary}
    return example


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_fn", type=str)
    parser.add_argument("--results_dir", type=str)
    parser.add_argument("--corpus", choices=SUPPORTED_CORPORA, default="tinystress")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data_root", type=Path, default=Path("data"))
    args = parser.parse_args()
    # Load your dataset, replace with the actual dataset you are using
    dataset_name = args.corpus
    split_name = args.split
    raw_dataset = load_corpus(args.corpus, args.split, args.data_root / "raw")
    
    print(f"Evaluating WhiStress on {dataset_name} for split {split_name}...")
    if args.metadata_fn is not None:
        with open(args.metadata_fn, "r") as fn:
            metadata = json.load(fn)
    else:
        metadata = None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    whistress_client = WhiStressInferenceClient(device=device, metadata=metadata)
    model = whistress_client.whistress
    #dataset[split_name] = dataset[split_name].map(add_stress_pattern, num_proc=4)
    #metrics, error_cases = calculate_metrics_on_dataset(dataset=dataset[split_name], whistress_client=whistress_client)
    #metrics_wot, error_cases_wot = calculate_metrics_on_dataset(dataset=dataset[split_name], whistress_client=whistress_client, with_transcription=False)
    processed_dir = args.data_root / "processed" / args.corpus / args.split
    dataset = StressDataset(hf_dataset_or_path=raw_dataset, model=model, processed_dir=str(processed_dir))
    metrics, metrics_wsd, error_cases, coverage = calculate_metrics_on_dataset(dataset=dataset, whistress_client=whistress_client, device=device)
    metrics_wot, _, error_cases_wot, coverage_wot = calculate_metrics_on_dataset(dataset=dataset, whistress_client=whistress_client, with_transcription=False, device=device)

    corpus_stats = {
        "num_original_samples": len(raw_dataset),
        "num_filtered_invalid_emphasis": 0,
        "num_retained_samples": len(raw_dataset),
    }
    if args.corpus in ("expresso", "emphassess"):
        corpus_stats = json.loads(raw_dataset.info.description)

    results = {
        "dataset": dataset_name,
        "split": split_name,
        "num_original_samples": corpus_stats["num_original_samples"],
        "num_filtered_invalid_emphasis": corpus_stats.get("num_filtered_invalid_emphasis", 0),
        "num_filtered_by_protocol": corpus_stats.get("num_filtered_by_protocol", 0),
        "num_samples": len(raw_dataset),
        "metrics": metrics,
        "metrics_wsd": ({
            **metrics_wsd,
            "label_source": "cmudict_g2p_lexical_stress",
        } if metrics_wsd is not None else None),
        "metrics_wot": metrics_wot,
        "coverage": coverage,
        "coverage_wot": coverage_wot,
    }

    # Save or log the metrics as needed
    if args.results_dir is None:
        results_dir = CURRENT_DIR / "evaluation_results"
    else:
        results_dir = Path(args.results_dir)
    
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(f"{results_dir}/whistress_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(f"{results_dir}/whistress_error_analysis.json", "w") as f:
        json.dump(error_cases, f, indent=2)
    
    with open(f"{results_dir}/whistress_error_analysis_wot.json", "w") as f:
        json.dump(error_cases_wot, f, indent=2)
        
