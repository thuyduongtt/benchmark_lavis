#!/bin/bash


LIMIT=20000
DS_NAME="ReasonVQA"
DS_VERSION="unbalanced"
TASK='vqa'

DS_DIR="../dataset/${DS_VERSION}"

case $1 in
  1)
    MODEL_NAME="blip2_t5_instruct"
    MODEL_TYPE="flant5xxl"
    START=0
    MULTICHOICE=true
    ;;
  2)
    MODEL_NAME="blip2_t5_instruct"
    MODEL_TYPE="flant5xxl"
    START=20000
    MULTICHOICE=true
    ;;
  3)
    MODEL_NAME="blip2_t5_instruct"
    MODEL_TYPE="flant5xxl"
    START=40000
    MULTICHOICE=true
    ;;
  4)
    MODEL_NAME="blip2_t5_instruct"
    MODEL_TYPE="flant5xxl"
    START=60000
    MULTICHOICE=true
    ;;
  5)
    MODEL_NAME="blip2_t5_instruct"
    MODEL_TYPE="flant5xxl"
    START=0
    MULTICHOICE=false
    ;;
  6)
    MODEL_NAME="blip2_t5_instruct"
    MODEL_TYPE="flant5xxl"
    START=20000
    MULTICHOICE=false
    ;;
  7)
    MODEL_NAME="blip2_t5_instruct"
    MODEL_TYPE="flant5xxl"
    START=40000
    MULTICHOICE=false
    ;;
  8)
    MODEL_NAME="blip2_t5_instruct"
    MODEL_TYPE="flant5xxl"
    START=60000
    MULTICHOICE=false
    ;;


  9)
    MODEL_NAME="blip2_t5"
    MODEL_TYPE="pretrain_flant5xl"
    START=0
    MULTICHOICE=true
    ;;
  10)
    MODEL_NAME="blip2_t5"
    MODEL_TYPE="pretrain_flant5xl"
    START=20000
    MULTICHOICE=true
    ;;
  11)
    MODEL_NAME="blip2_t5"
    MODEL_TYPE="pretrain_flant5xl"
    START=40000
    MULTICHOICE=true
    ;;
  12)
    MODEL_NAME="blip2_t5"
    MODEL_TYPE="pretrain_flant5xl"
    START=60000
    MULTICHOICE=true
    ;;
  13)
    MODEL_NAME="blip2_t5"
    MODEL_TYPE="pretrain_flant5xl"
    START=0
    MULTICHOICE=false
    ;;
  14)
    MODEL_NAME="blip2_t5"
    MODEL_TYPE="pretrain_flant5xl"
    START=20000
    MULTICHOICE=false
    ;;
  15)
    MODEL_NAME="blip2_t5"
    MODEL_TYPE="pretrain_flant5xl"
    START=40000
    MULTICHOICE=false
    ;;
  16)
    MODEL_NAME="blip2_t5"
    MODEL_TYPE="pretrain_flant5xl"
    START=60000
    MULTICHOICE=false
    ;;


  17)
    MODEL_NAME="blip2_opt"
    MODEL_TYPE="pretrain_opt6.7b"
    START=0
    MULTICHOICE=true
    ;;
  18)
    MODEL_NAME="blip2_opt"
    MODEL_TYPE="pretrain_opt6.7b"
    START=20000
    MULTICHOICE=true
    ;;
  19)
    MODEL_NAME="blip2_opt"
    MODEL_TYPE="pretrain_opt6.7b"
    START=40000
    MULTICHOICE=true
    ;;
  20)
    MODEL_NAME="blip2_opt"
    MODEL_TYPE="pretrain_opt6.7b"
    START=60000
    MULTICHOICE=true
    ;;
  21)
    MODEL_NAME="blip2_opt"
    MODEL_TYPE="pretrain_opt6.7b"
    START=0
    MULTICHOICE=false
    ;;
  22)
    MODEL_NAME="blip2_opt"
    MODEL_TYPE="pretrain_opt6.7b"
    START=20000
    MULTICHOICE=false
    ;;
  23)
    MODEL_NAME="blip2_opt"
    MODEL_TYPE="pretrain_opt6.7b"
    START=40000
    MULTICHOICE=false
    ;;
  24)
    MODEL_NAME="blip2_opt"
    MODEL_TYPE="pretrain_opt6.7b"
    START=60000
    MULTICHOICE=false
    ;;
esac


OUTPUT_NAME=${MODEL_NAME}_${MODEL_TYPE}_${DS_NAME}_${DS_VERSION}_${START}


if [ "$MULTICHOICE" = true ] ; then
  python start.py \
   --task $TASK \
   --ds_name $DS_NAME \
   --ds_dir $DS_DIR \
   --output_dir_name output_mc_${OUTPUT_NAME} \
   --model_name $MODEL_NAME \
   --model_type $MODEL_TYPE \
   --start_at $START \
   --limit $LIMIT \
   --multichoice
else
  python start.py \
   --task $TASK \
   --ds_name $DS_NAME \
   --ds_dir $DS_DIR \
   --output_dir_name output_${OUTPUT_NAME} \
   --model_name $MODEL_NAME \
   --model_type $MODEL_TYPE \
   --start_at $START \
   --limit $LIMIT
fi


