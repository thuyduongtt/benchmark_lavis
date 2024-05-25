#!/bin/bash

DS_NAME="OKVQA"
MODEL_NAME="blip2_t5_instruct"
MODEL_TYPE="flant5xxl"

case $1 in
  1)
    MULTICHOICE=true
    ;;
  2)
    MULTICHOICE=false
    ;;

DS_DIR="../dataset/${DS_NAME}"
IMG_DIR="../dataset/COCO/val2014"

if [ "$MULTICHOICE" = true ] ; then
  python start.py \
   --ds_name $DS_NAME \
   --ds_dir $DS_DIR \
   --img_dir $IMG_DIR \
   --output_dir_name output_mc_${MODEL_NAME}_${MODEL_TYPE}_${DS_NAME} \
   --model_name $MODEL_NAME \
   --model_type $MODEL_TYPE \
   --multichoice

else
  python start.py \
   --ds_name $DS_NAME \
   --ds_dir $DS_DIR \
   --img_dir $IMG_DIR \
   --output_dir_name output_${MODEL_NAME}_${MODEL_TYPE}_${DS_NAME} \
   --model_name $MODEL_NAME \
   --model_type $MODEL_TYPE
fi
