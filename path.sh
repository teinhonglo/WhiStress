export PYTHONNOUSERSITE=1
export WANDB_DISABLED=true
export WANDB_MODE=offline
#export PYTHONPATH="."

CUDA_DIR=/usr/local/cuda-11.6

if [ -d $CUDA_DIR ]; then
    export PATH=$CUDA_DIR/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_DIR/lib64:$LD_LIBRARY_PATH
fi

#eval "$(conda shell.bash hook)"
eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"
conda activate whistress
