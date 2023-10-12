#!/bin/bash

# Install gdown
pip install --upgrade --no-cache-dir gdown

# Download dataset - unbalanced
ZIP_ID='1MYcKvtQY5QCC1Cc3VSTDlyJgOFBBtdiN'
ZIP_NAME='unbalanced.zip'
gdown $ZIP_ID -O $ZIP_NAME
unzip -q $ZIP_NAME
rm $ZIP_NAME

# Download dataset - balanced 10
ZIP_ID='1MYcKvtQY5QCC1Cc3VSTDlyJgOFBBtdiN'
ZIP_NAME='balanced_10.zip'
gdown $ZIP_ID -O $ZIP_NAME
unzip -q $ZIP_NAME
rm $ZIP_NAME

