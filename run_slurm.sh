#!/bin/bash

hostname

script_root="$(dirname "$0")"
source $script_root/env.sh


RUNPATH=/storage/Python/guided-diffusion/
cd $RUNPATH
source /storage/Python/DiffusionModelTest/.venv/bin/activate

export RANK=${SLURM_NODEID}
#export MASTER_ADDR=192.168.1.37
export MASTER_ADDR=`getent hosts kubuntugpu | awk '{print $1}'`
export MASTER_PORT=29400
export WORLD_SIZE=${SLURM_NNODES}

#export OPENAI_LOGDIR="/storage/runs/run1"

#MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
#DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
#DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False"
#TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --microbatch 4"

#MODEL_FLAGS="--image_size 256 --num_channels 192 --num_res_blocks 3 --learn_sigma True --class_cond False"
#DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False"
#TRAIN_FLAGS="--lr 3e-4 --batch_size 1024 --microbatch 2"


echo "Running: python scripts/image_train.py --data_dir /storage/data/Dataset0622/ --resume_checkpoint $OPENAI_LOGDIR --save_interval 100 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS"

python scripts/image_train.py --data_dir /storage/data/Dataset0622/ --resume_checkpoint $OPENAI_LOGDIR --save_interval 100 $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

