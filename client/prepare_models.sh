#!/bin/bash
## One can rewrite this script with use of Python
## 'pathlib' and 'wget' packages.

set -e
extract_dir=${1:-configs}
## Better default would be models, but in Dockerfile, we will
## extract to a folder with config files of corresponding models.

## TensorFlow Inception
model=https://storage.googleapis.com/download.tensorflow.org
model+=/models/inception_v3_2016_08_28_frozen.pb.tar.gz
folder="$extract_dir/inception-graphdef"

mkdir -p "$folder/1"
wget -q -N -O inception.pb.tar.gz $model \
  || echo Failed to download from $model
tar xf inception.pb.tar.gz \
  --transform='s/.*frozen.pb/model.graphdef/' \
  --transform='s/.*labels.txt/inception_labels.txt/' \
  -C "$folder/1"

mv "$folder/1/inception_labels.txt" "$folder/"
rm inception.pb.tar.gz


## ONNX efficientnet-lite4
model=https://github.com/onnx/models/raw/main/vision/classification
model+=/efficientnet-lite4/model/efficientnet-lite4-11.onnx
folder="$extract_dir/efficientnet-lite4"

mkdir -p "$folder/1"
wget -q -N -O "$folder/1/model.onnx" $model \
  || echo Failed to download from $model

./pyscripts/enlite4_labels.py "$folder"


## QuartzNet15x5
## The commented lines is for extracting models from downloaded components.
# model=https://api.ngc.nvidia.com/v2/models/nvidia/quartznet15x5/versions/2/zip
# yaml=https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples
# yaml+=/asr/conf/quartznet/quartznet_15x5.yaml
folder="$extract_dir/quartznet15x5"

# wget -q --content-disposition ${model} -O quartznet15x5.zip
# ## ↓ `-n` ─ in case, a user executes the sciprt in a container ↓
# unzip -q -n quartznet15x5.zip

# wget -q ${yaml} --directory-prefix quartznet15x5  # default yaml is not OK
# ./pyscripts/quartznet15x5.py --output "$folder/1/model.onnx" quartznet15x5
# rm quartznet15x5.zip # quartznet15x5 -rf

## (1, 64, 1024) is the default input shape
./pyscripts/quartznet15x5.py \
  --model-name 'QuartzNet15x5Base-En' \
  --output "$folder/1/model.onnx" \
  --inp-shape '(1, 64, 1024)'
