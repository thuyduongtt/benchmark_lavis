#!/bin/bash

case $1 in
  1)
    DS_NAME="unbalanced"
    MODEL_NAME="blip_vqa"
    MODEL_TYPE="okvqa"
    ;;
  2)
    DS_NAME="balanced_10"
    MODEL_NAME="blip_vqa"
    MODEL_TYPE="okvqa"
    ;;
  3)
    DS_NAME="unbalanced"
    MODEL_NAME="blip2_opt"
    MODEL_TYPE="pretrain_opt6.7b"
    ;;
  4)
    DS_NAME="balanced_10"
    MODEL_NAME="blip2_opt"
    MODEL_TYPE="pretrain_opt6.7b"
    ;;
esac


DS_DIR="../dataset/${DS_NAME}"
python start.py --path_to_ds $DS_DIR --output_dir_name output_${DS_NAME} --model_name $MODEL_NAME --model_type $MODEL_TYPE
python start.py --path_to_ds $DS_DIR --output_dir_name output_${DS_NAME}_test --split test --model_name $MODEL_NAME --model_type $MODEL_TYPE
