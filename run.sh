#!/bin/bash
# dependency: torch, torchaudio, transformers, datasets, librosa

set -euo pipefail

stage=0
stop_stage=1000
train_conf=conf/baseline.json
gpuid=0

. ./local/parse_options.sh
. ./path.sh

exp_dir=exp/$(basename -s .json $train_conf)

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then

    if [ ! -f $exp_dir/.done ]; then
        CUDA_VISIBLE_DEVICES=$gpuid \
            python train.py \
                --train_conf $train_conf \
                --exp_dir $exp_dir
    fi
    touch $exp_dir/.done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    results_dir=$exp_dir/test

    mkdir -p $results_dir

    CUDA_VISIBLE_DEVICES=$gpuid \
        python test.py \
            --pretrained_ckpt_dir $exp_dir/best \
            --batch_size 1 \
            --exp_dir $exp_dir > $results_dir/test.log
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    metadata_fn=$exp_dir/best/metadata.json
    results_dir=$exp_dir/test

    echo "CUDA_VISIBLE_DEVICES=$gpuid python evaluation_example.py --metadata_fn $metadata_fn --results_dir $results_dir"

    CUDA_VISIBLE_DEVICES=$gpuid \
        python evaluation_example.py \
            --metadata_fn $metadata_fn \
            --results_dir $results_dir
    
    cat $results_dir/whistress_evaluation.json
    
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    metadata_fn=$exp_dir/best/metadata.json
    results_dir=$exp_dir/test
    
    python local/plot_evaluation_results.py \
        --error_case_path $results_dir/whistress_error_analysis.json \
        --save_fig_dir $results_dir/imgs
fi

