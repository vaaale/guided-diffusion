#!/bin/bash

export OPENAI_LOGDIR="/storage/runs/run1"
export OPENAI_LOG_FORMAT="stdout,log,csv,tensorboard"
export MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
#export DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
export DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --microbatch 4"

