import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

from e2e_stt.nlp_models import NlpModel

def plot_fnr_fpr_by(df, feature, save_dir, bins=None, labels=None):
    df_copy = df.copy()

    if bins:
        df_copy["bin"] = pd.cut(df_copy[feature], bins=bins, labels=labels, include_lowest=True)
    else:
        df_copy["bin"] = df_copy[feature]

    grouped = df_copy.groupby(["bin", "type"], observed=False).size().unstack(fill_value=0)

    grouped["FNR"] = grouped["FN"] / (grouped["FN"] + grouped["TP"]).replace(0, np.nan)
    grouped["FPR"] = grouped["FP"] / (grouped["FP"] + grouped["TN"]).replace(0, np.nan)

    plt.figure(figsize=(8, 5))
    grouped[["FNR", "FPR"]].plot(kind="bar", stacked=False, colormap="Set2", figsize=(8, 5))
    plt.title(f"FNR / FPR by {feature}")
    plt.xlabel(feature)
    plt.ylabel("Rate")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_path = save_dir / f"fnr_fpr_by_{feature}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved plot: {output_path}")

def plot_error_count_by(df, feature, save_dir, bins=None, labels=None):
    df_copy = df.copy()

    if bins:
        df_copy["bin"] = pd.cut(df_copy[feature], bins=bins, labels=labels, include_lowest=True)
    else:
        df_copy["bin"] = df_copy[feature]

    df_filtered = df_copy[df_copy["type"].isin(["FP", "FN"])]

    grouped = df_filtered.groupby(["bin", "type"], observed=False).size().reset_index(name="count")
    pivoted = grouped.pivot(index="bin", columns="type", values="count").fillna(0)

    plt.figure(figsize=(8, 5))
    pivoted.plot(kind="bar", stacked=False, colormap="Set2", figsize=(8, 5))
    plt.title(f"FP / FN Count by {feature}")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_path = save_dir / f"fp_fn_count_by_{feature}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved plot: {output_path}")

def flatten_words_with_pos(error_cases, nlp_model):
    rows = []
    for utt in error_cases:
        words = " ".join([w["word"] for w in utt["words"]])
        vp_feats = nlp_model.vocab_profile_feats(words)
        pos_feats = vp_feats["pos_list"]

        for i, word in enumerate(utt["words"]):
            rows.append({
                "word": word["word"],
                "gt": word["gt"],
                "pred": word["pred"],
                "type": word["type"],
                "word_len": word["word_len"],
                "syllable_count": word.get("syllable_count", None),
                "utt_len": utt["utt_len"],
                "utt_duration": utt["utt_duration"],
                "speaking_rate": utt["speaking_rate"],
                "pos": pos_feats[i]
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--error_case_path", type=str, default="evaluation_results/whistress_error_analysis.json", help="Path to whistress_error_analysis.json")
    parser.add_argument("--save_fig_dir", type=str, default="evaluation_results/figs", help="Directory to save figures")
    args = parser.parse_args()
    

    error_case_path = Path(args.error_case_path)
    save_fig_dir = Path(args.save_fig_dir)
    save_fig_dir.mkdir(parents=True, exist_ok=True)

    with open(error_case_path, "r") as f:
        error_cases = json.load(f)

    nlp_model = NlpModel()
    df = flatten_words_with_pos(error_cases, nlp_model)

    plot_error_count_by(df, "syllable_count", save_fig_dir)
    plot_error_count_by(df, "word_len", save_fig_dir, bins=[0, 3, 6, 10, 20], labels=["1-3", "4-6", "7-10", "11+"])
    plot_error_count_by(df, "speaking_rate", save_fig_dir, bins=[0, 2, 4, 6, 10], labels=["0-2", "2-4", "4-6", "6+"])

    plot_fnr_fpr_by(df, "syllable_count", save_fig_dir)
    plot_fnr_fpr_by(df, "word_len", save_fig_dir, bins=[0, 3, 6, 10, 20], labels=["1-3", "4-6", "7-10", "11+"])
    plot_fnr_fpr_by(df, "speaking_rate", save_fig_dir, bins=[0, 2, 4, 6, 10], labels=["0-2", "2-4", "4-6", "6+"])

    plot_error_count_by(df, "pos", save_fig_dir)
    plot_fnr_fpr_by(df, "pos", save_fig_dir)
