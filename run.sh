#!/bin/bash

case $1 in
  1)
    TASK='vqa'
    DS_NAME="unbalanced"
    MODEL_NAME="blip_vqa"
    MODEL_TYPE="okvqa"
    ;;
  2)
    TASK='vqa'
    DS_NAME="balanced_10"
    MODEL_NAME="blip_vqa"
    MODEL_TYPE="okvqa"
    ;;
  3)
    TASK='vqa'
    DS_NAME="unbalanced"
    MODEL_NAME="blip2_opt"
    MODEL_TYPE="pretrain_opt6.7b"
    ;;
  4)
    TASK='vqa'
    DS_NAME="balanced_10"
    MODEL_NAME="blip2_opt"
    MODEL_TYPE="pretrain_opt6.7b"
    ;;
  5)
    TASK='image_captioning'
    DS_NAME="unbalanced"
    MODEL_NAME="blip2_opt"
    MODEL_TYPE="caption_coco_opt6.7b"
    ;;
  6)
    TASK='image_captioning'
    DS_NAME="balanced_10"
    MODEL_NAME="blip2_opt"
    MODEL_TYPE="caption_coco_opt6.7b"
    ;;
  7)
    TASK='vqa'
    DS_NAME="unbalanced"
    MODEL_NAME="blip2_t5"
    MODEL_TYPE="pretrain_flant5xl"
    ;;
  8)
    TASK='vqa'
    DS_NAME="balanced_10"
    MODEL_NAME="blip2_t5"
    MODEL_TYPE="pretrain_flant5xl"
    ;;
  9)
    TASK='vqa'
    DS_NAME="unbalanced"
    MODEL_NAME="blip2_vicuna_instruct"
    MODEL_TYPE="vicuna13b"
    ;;
  10)
    TASK='vqa'
    DS_NAME="balanced_10"
    MODEL_NAME="blip2_vicuna_instruct"
    MODEL_TYPE="vicuna13b"
    ;;
  11)
    TASK='vqa'
    DS_NAME="unbalanced"
    MODEL_NAME="blip2_t5_instruct"
    MODEL_TYPE="flant5xxl"
    ;;
  12)
    TASK='vqa'
    DS_NAME="balanced_10"
    MODEL_NAME="blip2_t5_instruct"
    MODEL_TYPE="flant5xxl"
    ;;
esac

DS_DIR="../dataset/${DS_NAME}"
python start.py \
 --task $TASK \
 --path_to_ds $DS_DIR \
 --output_dir_name output_${MODEL_NAME}_${MODEL_TYPE}_${DS_NAME} \
 --model_name $MODEL_NAME \
 --model_type $MODEL_TYPE

python start.py \
 --task $TASK \
 --path_to_ds $DS_DIR \
 --output_dir_name output_${MODEL_NAME}_${MODEL_TYPE}_${DS_NAME}_test \
 --split test \
 --model_name $MODEL_NAME \
 --model_type $MODEL_TYPE
