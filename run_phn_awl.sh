#!/bin/bash
# dependency: torch, torchaudio, transformers, datasets, librosa

set -euo pipefail

stage=0
stop_stage=1000
# "precision": 0.8993174061433447, "recall": 0.9289071680376029, "f1": 0.9138728323699422 
init_lr=1e-4
epochs=20
batch_size=16
accumulate_gradient_steps=1
patience=-1
whisper_tag="openai/whisper-small.en"
model_type=wordstress
lambda_awl=1.0
gpuid=0

. ./local/parse_options.sh
. ./path.sh

model_tag=$(echo $whisper_tag | sed -e "s/\//-/g")
exp_dir=exp/${model_type}_${model_tag}_ep${epochs}_b${batch_size}a${accumulate_gradient_steps}_lr${init_lr}_p${patience}_awl$lambda_awl

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "CUDA_VISIBLE_DEVICES=$gpuid python train.py --whisper_tag $whisper_tag --epochs $epochs --init_lr ${init_lr} --batch_size $batch_size --exp_dir $exp_dir"

    if [ ! -f $exp_dir/.done ]; then
        CUDA_VISIBLE_DEVICES=$gpuid \
            python train_phn_awl.py \
                --whisper_tag $whisper_tag \
                --epochs $epochs \
                --init_lr ${init_lr} \
                --batch_size $batch_size \
                --accumulate_gradient_steps $accumulate_gradient_steps \
                --patience $patience \
                --lambda_awl $lambda_awl \
                --model_type $model_type \
                --exp_dir $exp_dir
    fi
    touch $exp_dir/.done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    results_dir=$exp_dir/test

    mkdir -p $results_dir
    echo "CUDA_VISIBLE_DEVICES=$gpuid python test.py --pretrained_ckpt_dir $exp_dir/best --whisper_tag $whisper_tag --init_lr ${init_lr} --batch_size $batch_size --exp_dir $exp_dir"

    CUDA_VISIBLE_DEVICES=$gpuid \
        python test.py \
            --pretrained_ckpt_dir $exp_dir/best \
            --whisper_tag $whisper_tag \
            --init_lr ${init_lr} \
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

