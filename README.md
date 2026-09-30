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

### POS bias

POS bias is applied to the SSD hidden states after the additional decoder block
and before the classifier. Two POS-bias modes are supported:

```bash
# Static POS residual
./run.sh --stage 1 --gpuid 0 --train_conf conf/baseline_pos_static.json

# Token-wise scalar-gated POS residual
./run.sh --stage 1 --gpuid 0 --train_conf conf/baseline_pos_scalar_gated.json
```

The `scalar_gated` mode computes one scalar gate per token and shares that
value across all hidden dimensions:

```text
g_t = sigmoid((W_q e_t)^T (W_k h_t) / sqrt(d_g) + b_g)
h'_t = h_t + mask_t * g_t * LayerNorm(W_v e_t)
```

Unlike the static mode, which uses a learnable residual scale, the scalar-gated
mode uses the token-level gate itself to control the POS contribution.
`gate_init: 0.01` initializes the POS contribution conservatively while
keeping the gating path trainable.

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

## ProWhistress reproduction

`ProWhiStress` implements the English dual-stream model from
[Gu et al., Interspeech 2026](https://www.isca-archive.org/interspeech_2026/gu26b_interspeech.pdf),
following the [official implementation](https://github.com/Guhujian/ProWhistress)
at commit `a41cd0e13bf39ebd50c46cdfecd49343d89708af`. It is an SSD model;
the existing WSD, POS, and coupling variants remain separate model types.

Start with the paper configuration:

```bash
./run.sh --stage 0 --stop_stage 4 --gpuid 0 \
  --train_conf conf/prowhistress_paper.json

# Data already downloaded: train and run the existing evaluation/plot stages.
./run.sh --stage 1 --stop_stage 4 --gpuid 0 \
  --train_conf conf/prowhistress_paper.json

# Evaluate a trained checkpoint again.
./run.sh --stage 2 --stop_stage 4 --gpuid 0 \
  --train_conf conf/prowhistress_paper.json
```

| Setting | `prowhistress_paper.json` | Source |
|---|---|---|
| Frozen backbone | `openai/whisper-small.en` | Paper, Section 4.3 |
| Decoder hidden state | 9 | Paper, Section 4.3 |
| Implicit stream encoder hidden state | 12 | Paper, Section 4.3 |
| Explicit stream encoder hidden state | 9 | Paper, Section 4.3 |
| Acoustic encoder | 3 Transformer layers, FFN width 3072, 12 heads | Paper + English model code |
| Bottleneck | 256 dimensions, 8 attention heads | Paper + code's head-divisibility fallback |
| Gate | Per-feature two-layer MLP; final bias -3 | Paper + model code |
| Acoustic encoder internal/output dropout | 0.1 / 0.0 | Model code / README training command |
| Positive CE weight | `0.7 / 0.3` | Model code; rounded to 2.33 in paper |
| Explicit-output regularization | 0 | README training command |
| Epochs / train batch / accumulation | 2 / 32 / 1 | Paper + README |
| Optimizer | Torch AdamW, betas (0.9, 0.999), epsilon 1e-8 | Author's Trainer defaults |
| Learning rate / weight decay | 5e-4 / 0.01 | Author's training code |
| Schedule / warmup | Linear decay / 5% | Author's training code and Trainer default |
| Validation split | 2% of TinyStress training split; fixed split seed 42 | Author's data loader |
| Validation / periodic checkpoint | Every 10 / 100 optimizer steps | Author's training code |
| Model seed | 42 initially; paper uses 42-46 | Paper, Section 4.3 |
| Gradient clipping / precision | Max norm 1.0 / FP32 | Author's Trainer settings/defaults |
| Audio-only generation length | 96 | Author's training code |

Layer indices refer to Hugging Face `hidden_states`, where index 0 is the
embedding output. The explicit attention uses the original decoder layer-9
state as its query, not the additional decoder's output. Both streams are
trained by stress cross-entropy; no ASR loss or handcrafted pitch features are
added. Training, transcript-conditioned inference, and audio-only inference
share the same stress head.

Checkpoints preserve both streams in `best/model.pt` and the complete
architecture in `best/metadata.json`. The highest validation token-level F1 is
saved immediately; periodic checkpoints retain only the latest file. A final
validation/checkpoint is also made when a short run ends between intervals.
Stage 2 still reports token-level metrics, Stage 3 word-level metrics and
coverage (with and without reference transcription), and Stage 4 the existing
error-analysis PNGs. In particular, compare the author's teacher-forced
word-level results against Stage 3 `metrics`, rather than `metrics_wot`.

To repeat all five model seeds using the same data split:

```bash
for seed in 42 43 44 45 46; do
  ./run.sh --stage 1 --stop_stage 4 --gpuid 0 \
    --train_conf conf/prowhistress_paper.json --seed "$seed"
done
```

`--seed` creates separate `exp/prowhistress_paper_seed<seed>` directories;
without it, the default is `exp/prowhistress_paper`. `--exp_dir` can override
this path explicitly. Split fingerprints and the text-length limit isolate
the paper's preprocessing caches from legacy 10%-validation/50-token caches.

`conf/prowhistress.json` uses the same dual-stream architecture with this
project's original 20-epoch, batch-16, learning-rate-1e-4 training setup for
project comparisons. Use `prowhistress_paper.json` first for paper reproduction.

Reproduction qualifications:

- The published English model references an uninitialized `source_layer_idx`
  and an undefined `output_dim`, and its training entrypoint has an indentation
  error. The additional decoder is initialized from the last Whisper decoder
  block, following the corresponding Chinese code (`12` clamped to index `11`).
  This is a documented reconstruction of the missing English initialization.
- The implementation uses this project's word/token label mapping and audio
  preprocessing. It retains full transcripts up to Whisper's 448-token limit
  for ProWhistress instead of the legacy 50-token limit. The author's mapping,
  punctuation handling, and audio resampling differ, so numeric equivalence to
  the paper must be assessed empirically. Whisper-mismatch filtering is
  commented out in the published English preprocessing and is not added here.
- Standard `train()`/`eval()` behavior is preserved for the entire acoustic
  branch, including dropout. The author's model overrides omit this recursive
  handling, which can leave output dropout active during evaluation.
- SinoStress data construction, Mandarin evaluation, and supervised
  EmphAssess training are outside this English/TinyStress reproduction. Expresso
  and EmphAssess are evaluated zero-shot through the existing adapters.

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
