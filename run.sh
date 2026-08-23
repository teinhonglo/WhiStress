#!/bin/bash
# dependency: torch, torchaudio, transformers, datasets, librosa

set -euo pipefail

stage=0
stop_stage=1000
train_conf=conf/baseline.json
gpuid=0
data_root=data
test_corpora="tinystress stresstest stresspresso emphassess"
force_download=false

. ./local/parse_options.sh
. ./path.sh

exp_dir="exp/$(basename -s .json "$train_conf")"

if [ "$stage" -le 0 ] && [ "$stop_stage" -ge 0 ]; then
    download_args=(
        --data_root "$data_root/raw"
        --corpora $test_corpora
    )
    if [ "$force_download" = true ]; then
        download_args+=(--force)
    fi
    python local/download_corpora.py "${download_args[@]}"
fi

if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 1 ]; then

    if [ ! -f "$exp_dir/.done" ]; then
        CUDA_VISIBLE_DEVICES="$gpuid" \
            python train.py \
                --train_conf "$train_conf" \
                --exp_dir "$exp_dir"
    fi
    touch "$exp_dir/.done"
fi

if [ "$stage" -le 2 ] && [ "$stop_stage" -ge 2 ]; then
    for corpus in $test_corpora; do
        results_dir="$exp_dir/test/$corpus"
        mkdir -p "$results_dir"
        CUDA_VISIBLE_DEVICES="$gpuid" python test.py \
            --pretrained_ckpt_dir "$exp_dir/best" --batch_size 1 \
            --exp_dir "$exp_dir" --corpus "$corpus" --split test \
            --data_root "$data_root" --results_dir "$results_dir" \
            > "$results_dir/stage2.log"
    done
fi

if [ "$stage" -le 3 ] && [ "$stop_stage" -ge 3 ]; then
    metadata_fn="$exp_dir/best/metadata.json"
    for corpus in $test_corpora; do
        results_dir="$exp_dir/test/$corpus"
        mkdir -p "$results_dir"
        CUDA_VISIBLE_DEVICES="$gpuid" python evaluation_example.py \
            --metadata_fn "$metadata_fn" --corpus "$corpus" --split test \
            --data_root "$data_root" --results_dir "$results_dir" \
            > "$results_dir/stage3.log"
        cat "$results_dir/whistress_evaluation.json"
    done
fi

if [ "$stage" -le 4 ] && [ "$stop_stage" -ge 4 ]; then
    for corpus in $test_corpora; do
        results_dir="$exp_dir/test/$corpus"
        python local/plot_evaluation_results.py \
            --error_case_path "$results_dir/whistress_error_analysis.json" \
            --save_fig_dir "$results_dir/imgs"
    done
fi
