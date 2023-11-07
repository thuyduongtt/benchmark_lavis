#!/bin/bash

conda activate lavis

DS_NAME="unbalanced"
DS_DIR="../dataset/${DS_NAME}"
python start.py --path_to_ds $DS_DIR --output_dir_name output_${DS_NAME}
python start.py --path_to_ds $DS_DIR --output_dir_name output_${DS_NAME}_test --split test