#!/bin/bash
Fairseq_path=./fairseq
GPUS=0,1,2,3
num_gpus=$(echo "$GPUS" | tr -cd ',' | wc -c)
num_gpus=$((num_gpus + 1))

CUDA_VISIBLE_DEVICES=$GPUS \
PYTHONPATH=$Fairseq_path \
torchrun --standalone --nnodes=1 --nproc_per_node=${num_gpus} \
train.py \
--coco_path path_to/COCO_2014 \
--flickr_path path_to/Flickr8k/Images \
--spcoco_path path_to/SpokenCOCO/Hubert_units \
--spflickr_path path_to/flickr_audio/Hubert_units \
--spcoco_split_path path_to/SpokenCOCO \
--karpathy_split_path path_to/Karpathy_split \
--checkpoint_dir ./data/checkpoints/TMT \
--temp_dir ./tmp_eval/TMT \
--batch_size 20 \
--eval_step 5000 \
--lr 1e-5 \
--gpu $GPUS \
--update_frequency 1 \
--train_data coco \
--image_tokenizer SEED \
--generation_step 1000 \
--architecture bert \
--fp16 \
--im_txt \
--im_sp \
--txt_sp \
--txt_im \
--sp_txt \
--sp_im \
--num_task 6 \
--warmup \
--warmup_iteration 10000 \
--tot_iters 50000 \
--masterport 8686 \
--distributed "$@"

#--checkpoint ./data/checkpoints/xx.ckpt    # to resume from
#--project TMT    # for wandb logging
#--generate_im  # whether to generate images during validation (super slow)



