#!/bin/bash

set -euo pipefail

stage=1
stop_stage=2
train_conf=conf/baseline.json
gpuid=0
data_root=data
test_corpora="tinystress"
alpha=1.0
plausibility_alpha=0.0
contrast_scope=all_steps
max_new_tokens=128
max_samples=0
dtype=auto
pretrained_ckpt_dir=
help_message="Usage: $0 [options]
  --stage INT --stop-stage INT --gpuid INT
  --train-conf PATH --pretrained-ckpt-dir PATH
  --test-corpora STRING --alpha FLOAT
  --contrast-scope first_step|all_steps|both
  --plausibility-alpha FLOAT --max-new-tokens INT
  --max-samples INT --dtype auto|float16|bfloat16|float32"

. ./local/parse_options.sh
. ./path.sh

exp_dir="exp/$(basename -s .json "$train_conf")"
if [ -z "$pretrained_ckpt_dir" ]; then
    pretrained_ckpt_dir="$exp_dir/best"
fi

if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 1 ]; then
    if [ ! -f "$pretrained_ckpt_dir/metadata.json" ]; then
        echo "Missing checkpoint metadata: $pretrained_ckpt_dir/metadata.json" >&2
        exit 1
    fi
    for corpus in $test_corpora; do
        results_dir="$exp_dir/test/$corpus/silence_contrast/scope_${contrast_scope}_alpha_${alpha}_plausibility_${plausibility_alpha}"
        mkdir -p "$results_dir"
        CUDA_VISIBLE_DEVICES="$gpuid" python silence_contrast.py \
            --mode decode \
            --pretrained_ckpt_dir "$pretrained_ckpt_dir" \
            --corpus "$corpus" \
            --split test \
            --data_root "$data_root" \
            --results_dir "$results_dir" \
            --alpha "$alpha" \
            --plausibility_alpha "$plausibility_alpha" \
            --contrast_scope "$contrast_scope" \
            --max_new_tokens "$max_new_tokens" \
            --max_samples "$max_samples" \
            --dtype "$dtype" \
            > "$results_dir/stage1.log" 2>&1
    done
fi

if [ "$stage" -le 2 ] && [ "$stop_stage" -ge 2 ]; then
    for corpus in $test_corpora; do
        results_dir="$exp_dir/test/$corpus/silence_contrast/scope_${contrast_scope}_alpha_${alpha}_plausibility_${plausibility_alpha}"
        CUDA_VISIBLE_DEVICES="$gpuid" python silence_contrast.py \
            --mode evaluate \
            --pretrained_ckpt_dir "$pretrained_ckpt_dir" \
            --corpus "$corpus" \
            --data_root "$data_root" \
            --results_dir "$results_dir" \
            > "$results_dir/stage2.log" 2>&1
        cat "$results_dir/silence_contrast_metrics.json"
    done
fi
