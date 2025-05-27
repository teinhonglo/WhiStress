# ***WhiStress***: Enriching Transcriptions with Sentence Stress Detection

The official repo of
["***WhiStress***: Enriching Transcriptions with Sentence Stress Detection"](https://arxiv.org/abs/2505.19103) (Interspeech 2025).
***WhiStress*** extends OpenAI’s [Whisper](https://arxiv.org/abs/2212.04356) to provide not only accurate transcriptions of speech, but also **token-level sentence stress annotations** — allowing you to detect which words are emphasized in a spoken sentence.

The model is built on top of the [`whisper-small.en`](https://huggingface.co/openai/whisper-small.en) variant, and enhanced with a lightweight decoder-based classifier that predicts the **stress label for each token**.

Checkout our [project's page](https://pages.cs.huji.ac.il/adiyoss-lab/whistress/) for more information.

## 🔧 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/slp-rl/WhiStress.git
cd WhiStress
pip install -r requirements.txt
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
source path/to/your/venv/bin/activate
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

Coming soon...


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
