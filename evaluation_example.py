import torch
import evaluate
from tqdm import tqdm
from pathlib import Path
from whistress import WhiStressInferenceClient
import pprint
import pyphen
import argparse

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


def calculate_metrics_on_dataset(dataset, whistress_client):
    """
    Sample structure example for slp-rl/StressTest dataset:
    # {'transcription_id': '98dd4a46-6b59-4817-befc-e35d548465c7',
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
    error_cases = []

    for sample in tqdm(dataset):
        gt_stresses = sample['stress_pattern']['binary']
        # scored = 
        scored = whistress_client.predict(
            audio=sample['audio'],
            # Using ground truth transcription for evaluating stress prediction ability. 
            # set transcription to None if not available
            transcription=sample['transcription'], 
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

        error_cases.append({
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

    metrics = compute_prf_metrics(predictions, references, average="binary")
    return metrics, error_cases

def compute_stress_binary(transcription: str, emphasis_indices: list[int]) -> list[int]:
    words = transcription.strip().split()
    return [1 if i in emphasis_indices else 0 for i in range(len(words))]

def add_stress_pattern(example):
    binary = compute_stress_binary(example["transcription"], example["emphasis_indices"])
    example["stress_pattern"] = {"binary": binary}
    return example


if __name__ == "__main__":
    from datasets import load_dataset
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_fn", type=str)
    parser.add_argument("--results_dir", type=str)
    args = parser.parse_args()
    # Load your dataset, replace with the actual dataset you are using
    dataset_name = "slprl/TinyStress-15K"  # Example dataset name, change as needed
    dataset = load_dataset(dataset_name)
    split_name = 'test' 
    
    print(f"Evaluating WhiStress on {dataset_name} for split {split_name}...")
    if args.metadata_fn is not None:
        with open(args.metadata_fn, "r") as fn:
            metadata = json.load(fn)
    else:
        metadata = None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    whistress_client = WhiStressInferenceClient(device=device, metadata=metadata)
    dataset[split_name] = dataset[split_name].map(add_stress_pattern, num_proc=4)
    
    metrics, error_cases = calculate_metrics_on_dataset(dataset=dataset[split_name], whistress_client=whistress_client)
    # Save or log the metrics as needed
    results = {
        "dataset": dataset_name,
        "split": split_name,
        "metrics": metrics
    }
    pprint.pp(f"Results: {results}")
    
    if args.results_dir is None:
        results_dir = CURRENT_DIR / "evaluation_results"
    else:
        results_dir = Path(args.results_dir)
    
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(f"{results_dir}/whistress_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(f"{results_dir}/whistress_error_analysis.json", "w") as f:
        json.dump(error_cases, f, indent=2)
        
