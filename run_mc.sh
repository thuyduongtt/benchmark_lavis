#!/bin/bash

LIMIT=20000
DS_NAME="unbalanced"
TASK='vqa'
MODEL_NAME="blip2_t5_instruct"
MODEL_TYPE="flant5xxl"

case $1 in
  1)
    START=0
    ;;
  2)
    START=20000
    ;;
  3)
    START=40000
    ;;
  4)
    START=60000
    ;;
  5)
    START=80000
    ;;
  6)
    START=100000
    ;;
  7)
    START=120000
    ;;
esac

DS_DIR="../dataset/${DS_NAME}"

python start.py \
 --task $TASK \
 --ds_name $DS_NAME \
 --ds_dir $DS_DIR \
 --output_dir_name output_mc_${MODEL_NAME}_${MODEL_TYPE}_${DS_NAME}_${START} \
 --model_name $MODEL_NAME \
 --model_type $MODEL_TYPE \
 --start_at $START \
 --limit $LIMIT \
 --multichoice

python start.py \
 --task $TASK \
 --ds_name $DS_NAME \
 --ds_dir $DS_DIR \
 --output_dir_name output_mc_${MODEL_NAME}_${MODEL_TYPE}_${DS_NAME}_test_${START} \
 --split test \
 --model_name $MODEL_NAME \
 --model_type $MODEL_TYPE \
 --start_at $START \
 --limit $LIMIT \
 --multichoice
