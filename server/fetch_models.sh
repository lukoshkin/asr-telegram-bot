#!/bin/bash

set -e
extract_dir=${1:-models}

## TensorFlow Inception
model=https://storage.googleapis.com/download.tensorflow.org/models/inception_v2_2016_08_28_frozen.pb.tar.gz
folder="$extract_dir/inception_graphdef"

mkdir -p "$folder/1"
wget -q -N -O inception.pb.tar.gz $model \
  || echo Failed to download from $model
tar xf inception.pb.tar.gz \
  --transform='s/inception.*frozen.pb/model.graphdef/' \
  --transform='s/.*labels.txt/inception_labels.txt/' \
  -C "$folder/1"

mv "$folder/1/inception_labels.txt" "$folder/"
rm inception.pb.tar.gz

## ONNX efficientnet-lite4
model=https://github.com/onnx/models/raw/main/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx
folder="$extract_dir/efficientnet_lite4"

mkdir -p "$folder/1"
wget -q -N -O "$folder/1/model.onnx" $model \
  || echo Failed to download from $model
