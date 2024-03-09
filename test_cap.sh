#!/bin/bash
Fairseq_path=./fairseq

################### SET ##################
SAVENAME=TMT_test_caption
CKPT=./data/checkpoints/xx.ckpt
DEVICE=0

# Data
COCO=path_to/COCO_2014
FLICKR=path_to/Flickr8k/Images
SPCOCO_U=path_to/SpokenCOCO/Hubert_units
SPFLICKR=path_to/flickr_audio/Hubert_units
SPCOCO=path_to/SpokenCOCO
KARPATHY=path_to/Karpathy_split
##########################################

REF=$COCO

## test for image captioning, image-to-speech captioning
CUDA_VISIBLE_DEVICES=$DEVICE \
PYTHONPATH=$Fairseq_path \
python test.py \
--coco_path $COCO \
--flickr_path $FLICKR \
--spcoco_path $SPCOCO_U \
--spflickr_path $SPFLICKR \
--spcoco_split_path $SPCOCO \
--karpathy_split_path $KARPATHY \
--im_txt \
--im_sp \
--test_data coco \
--beam_size 5 \
--batch_size 8 \
--gpu $DEVICE \
--image_tokenizer SEED \
--architecture bert \
--checkpoint ${CKPT} \
--save_name ${SAVENAME}

python score.py \
--pred test_results/${SAVENAME}/is_transcription \
--ref ${REF} 

python score.py \
--pred test_results/${SAVENAME}/it_transcription \
--ref ${REF} 

