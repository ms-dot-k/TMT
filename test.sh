#!/bin/bash
Fairseq_path=./fairseq

################### SET ##################
SAVENAME=TMT_test
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

## test for image captioning, image-to-speech captioning, text-to-image synthesis, speech-to-image synthesis
CUDA_VISIBLE_DEVICES=$DEVICE \
PYTHONPATH=${Fairseq_path} \
python test.py \
--coco_path $COCO \
--flickr_path $FLICKR \
--spcoco_path $SPCOCO_U \
--spflickr_path $SPFLICKR \
--spcoco_split_path $SPCOCO \
--karpathy_split_path $KARPATHY \
--txt_im \
--sp_im \
--im_txt \
--im_sp \
--test_data coco \
--beam_size 5 \
--gpu $DEVICE \
--batch_size 8 \
--generate_im \
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

CUDA_VISIBLE_DEVICES=$DEVICE \
python test_gen.py \
--pred ./test_results/${SAVENAME}/si_images \
--coco_path $COCO \
--flickr_path $FLICKR \
--spcoco_split_path $SPCOCO \
--karpathy_split_path $KARPATHY \
--data coco

CUDA_VISIBLE_DEVICES=$DEVICE \
python test_gen.py \
--pred ./test_results/${SAVENAME}/ti_images \
--coco_path $COCO \
--flickr_path $FLICKR \
--spcoco_split_path $SPCOCO \
--karpathy_split_path $KARPATHY \
--data coco

## test for automatic speech recognition, text-to-speech synthesis
CUDA_VISIBLE_DEVICES=$DEVICE \
PYTHONPATH=${Fairseq_path} \
python test_asr.py \
--coco_path $COCO \
--flickr_path $FLICKR \
--spcoco_path $SPCOCO_U \
--spflickr_path $SPFLICKR \
--spcoco_split_path $SPCOCO \
--karpathy_split_path $KARPATHY \
--sp_txt \
--txt_sp \
--test_data coco \
--beam_size 5 \
--gpu $DEVICE \
--batch_size 32 \
--image_tokenizer SEED \
--checkpoint ${CKPT} \
--save_name ${SAVENAME}
