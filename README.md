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

If you are interested in the official version, please refer to the original repository and [project page](https://pages.cs.huji.ac.il/adiyoss-lab/whistress/).

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

## Training

```bash
./run.sh --stage 1 --gpuid 0
```

## 📊 Results

| Name                  | Precision | Recall | F1    |
|-----------------------|-----------|--------|-------|
| Paper                 | 91.20     | 90.60  | 90.90 |
| Dry Run               | 88.84     | 93.31  | 91.02 |
| └─ without transcription | 88.15     | 94.17  | 91.06 |
| RP (Reproduced)       | 92.37     | 93.17  | 92.77 |
| └─ without transcription | 89.21     | 93.96  | 91.52 |

- **Paper**: Results reported in the original WhiStress paper.  
- **Dry Run**: Inference using the official pretrained weights without any retraining.  
- **RP (Reproduced)**: Results from retraining the model using the provided `model.py` and corpus.  
- *without transcription*: Evaluation conducted without using ground-truth transcriptions (i.e., `with_transcription=False` in `calculate_metrics_on_dataset`).

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
