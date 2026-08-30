## ⚠️ Notice on Modifications

This repository is an **unofficial extension** of the original repository
[WhiStress](https://github.com/slp-rl/WhiStress) (Interspeech 2025).

In addition to the original functionalities, this version includes:
- Added: `train.py`, `test.py`, and `run.sh` for streamlined training and evaluation
- Light modifications to:
  - `whistress/inference_client/utils.py`
  - `whistress/inference_client/whistress_client.py`
  - `whistress/model/model.py`
  - `evaluation_example.py`

These changes are intended to support custom workflows and reproducibility, while preserving alignment with the original implementation.

If you are interested in the official version, please refer to the [original repository](https://github.com/slp-rl/WhiStress) and [project page](https://pages.cs.huji.ac.il/adiyoss-lab/whistress/).

## 🔧 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/teinhonglo/WhiStress.git
cd WhiStress

# Create and activate the conda environment
conda create -n whistress python==3.10
conda activate whistress

# Install required packages
pip install -r requirements.txt
````

### Configure the Conda Environment

Modify the conda startup method in `path.sh` to match your own environment path:

```bash
vim path.sh
```

### Basic Version

```bash
export PYTHONNOUSERSITE=1

eval "$(conda shell.bash hook)"
conda activate whistress
```

## 📦 Model Weights

Download the model weights from [***WhiStress***](https://huggingface.co/slprl/WhiStress) 🤗 huggingface:
```
https://huggingface.co/slprl/WhiStress/tree/main
```
and place them inside the `whistress/weights` directory.

Expected structure:

```
whistress/
├── weights/
│   └── additional_decoder_block.pt
│   └── classifier.pt
│   └── metadata.json
├── ...
README.md
download_weights.py
...
```

You can use the `download_weights.py` script places under the main repo folder. 


## 📚 Training Data

WhiStress was trained on the [***TinyStress-15K***](https://huggingface.co/datasets/slprl/TinyStress-15K) dataset. This dataset is based on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), adapted for sentence stress supervision.


## 🚀 Usage

### 1. Activate environment

```bash
. ./path.sh
```

### 2. Run inference

To generate a transcription with stress predictions:

```bash
python inference_example.py
```

### 3. Evaluate the model

Run evaluation on a sample dataset:

```bash
python evaluation_example.py
```

## 🖥️ Demo UI

You can check out our [***Demo***](https://huggingface.co/datasets/loud-whisper-project/tinyStories-audio-emphasized) on 🤗 huggingface.

Or, run the interface locally:

```bash
python app_ui.py
```

This will launch a browser-based UI for trying out the model interactively on your own audio.

## 🏋️‍♀️ Training

```bash
# Baseline
./run.sh --stage 1 --gpuid 0 --train_conf conf/baseline.json

# Baseline + WSL
./run.sh --stage 1 --gpuid 0 --train_conf conf/baseline_wsl.json

# SSD + WSD
./run.sh --stage 1 --gpuid 0 --train_conf conf/wordstress.json

# SSD + WSD + WSL
./run.sh --stage 1 --gpuid 0 --train_conf conf/wordstress_wsl.json
```

### POS-bias injection comparison

The existing `conf/*pos*.json` configurations inject POS bias after the
additional decoder block. The corresponding `*_pre_additional*.json`
configurations inject the same POS bias into the selected Whisper decoder
hidden states before the additional decoder block. For example:

```bash
# Current: POS bias after the additional decoder
./run.sh --stage 1 --gpuid 0 --train_conf conf/baseline_pos_gated.json

# Comparison: POS bias before the additional decoder
./run.sh --stage 1 --gpuid 0 --train_conf conf/baseline_pos_gated_pre_additional.json
```

Older POS configs and checkpoints without an explicit `injection_point`
remain compatible and default to `after_additional_decoder`.

## 📊 Results

| Name                  | Precision | Recall | F1    |
|-----------------------|-----------|--------|-------|
| Paper                 | 91.20     | 90.60  | 90.90 |
| Dry Run               | 88.84     | 93.31  | 91.02 |
| └─ without transcription | 88.15     | 94.17  | 91.06 |
| RP       | 92.37     | 93.17  | 92.77 |
| └─ without transcription | 89.21     | 93.96  | 91.52 |

- **Paper**: Results reported in the original WhiStress paper.  
- **Dry Run**: Inference using the official pretrained weights without any retraining.  
- **RP**: Results from retraining the model using the provided `model.py` and corpus.  
- *without transcription*: Evaluation conducted without using ground-truth transcriptions (i.e., `with_transcription=False` in `calculate_metrics_on_dataset`[Link](https://github.com/teinhonglo/WhiStress/blob/main/evaluation_example.py#L79-L84)).

## Citation

If you use ***WhiStress*** in your work, please cite our paper:

```bibtex
@misc{yosha2025whistress,
    title={WHISTRESS: Enriching Transcriptions with Sentence Stress Detection}, 
    author={Iddo Yosha and Dorin Shteyman and Yossi Adi},
    year={2025},
    eprint={2505.19103},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2505.19103}, 
}
```

## Multi-corpus evaluation

The evaluation pipeline supports `tinystress`, `stresstest`, `stresspresso`,
`expresso`, and `emphassess`. Stage 0 downloads and validates them without
changing the Stage 1 training data or training procedure:

```bash
python local/download_corpora.py \
  --data_root data/raw \
  --corpora tinystress stresstest stresspresso expresso emphassess

# Download, train, test, evaluate, and plot.
./run.sh --stage 0 --stop_stage 4 --gpuid 0 --train_conf conf/baseline.json

# Reuse downloaded data and an existing checkpoint; run Stage 2 and Stage 3 only.
./run.sh --stage 2 --stop_stage 3 --gpuid 0 --train_conf conf/baseline.json

# Evaluate a subset (a quoted, space-separated parse_options.sh value).
./run.sh --stage 2 --stop_stage 3 --test_corpora "stresstest emphassess"
```

TinyStress-15K already supplies `transcription`, audio, and
`emphasis_indices`. StressTest and StressPresso supply interpretation-specific
IDs and nested `stress_pattern` labels; their binary labels are validated during
adaptation. Expresso follows the SSD protocol used by WhiStress and StressTest:
the `read` configuration is restricted to speakers `ex01` and `ex02`, and only
samples containing at least one asterisk-marked emphasis span are retained. The
asterisks are removed from the transcription and every word in each marked span
is mapped to `emphasis_indices`. Expresso exposes this material in its single
source `train` split, but it is used only as a held-out evaluation corpus here.
EmphAssess supplies token lists and `gold_emphasis`; its original 16-kHz source
WAV (not an output of the official speech-to-speech emphasis transfer pipeline)
is evaluated directly. All adapters produce this canonical shape before the
existing preprocessing runs:

```python
{
    "id": str,
    "transcription": str,
    "audio": {"array": ..., "sampling_rate": int, "path": str | None},
    "emphasis_indices": list[int],
    "source_dataset": str,
}
```

For EmphAssess, standalone punctuation is removed while apostrophes inside words
are retained, and emphasis indices are remapped. The 12 rows whose emphasis
points to standalone punctuation are rejected as invalid (3,652 original, 3,640
retained); labels are never shifted to a neighboring word. Expresso and
EmphAssess are distributed under **CC BY-NC 4.0**. Review the respective dataset
licenses before use.

Stage 2 reports teacher-forced **Whisper token-level** metrics. Stage 3 preserves
the inference evaluation's merged **word-level** metrics, both with and without
ground-truth transcription. Interpret without-transcription results together
with their separately reported coverage because word-length mismatches remain
skipped rather than being force-aligned.

Corpus-specific outputs are written to:

```text
exp/<config>/test/tinystress/
exp/<config>/test/stresstest/
exp/<config>/test/stresspresso/
exp/<config>/test/expresso/
exp/<config>/test/emphassess/
```
