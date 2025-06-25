#!/bin/bash
# dependency: torch, torchaudio, transformers, datasets, librosa

set -euo pipefail

stage=0
stop_stage=0
# "precision": 0.8993174061433447, "recall": 0.9289071680376029, "f1": 0.9138728323699422 
init_lr=1e-4
epochs=30
batch_size=8
gpuid=0

. ./local/parse_options.sh
. ./path.sh

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    exp_dir=exp/baseline_ep${epochs}_b${batch_size}_lr${init_lr}
    CUDA_VISIBLE_DEVICES=$gpuid \
        python train.py \
            --epochs $epochs \
            --init_lr ${init_lr} \
            --batch_size $batch_size \
            --exp_dir $exp_dir
fi
