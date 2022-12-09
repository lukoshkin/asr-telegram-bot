# syntax = docker/dockerfile:1.3
## ↑ Required is using Docker's BuildKit ↑

ARG IMAGE_TAG=21.09-py3
FROM nvcr.io/nvidia/tritonserver:${IMAGE_TAG} AS base

FROM base as dev
ARG host=https://api.ngc.nvidia.com
ARG path=/v2/models/nvidia/quartznet15x5/versions/2/zip
## Splitting in two to fit 80 chars width, no additional meaning.
ARG repo=https://raw.githubusercontent.com
ARG yaml_file=NVIDIA/NeMo/main/examples/asr/conf/quartznet/quartznet_15x5.yaml

COPY models prepare.py fetch_models.sh ./
ARG asr_model_dir=/models/quartznet15x5/1

## BuildKit allows setting `--mount` option of `RUN` instruction.
## Mounts don't help (speed up building) if the layer is cached. But here,
## they are given as a usage example. Note: you shouldn't delete what you are
## going to mount to the host. E.g., if `target=/root/.cache/pip`, then
## don't specify the option `--no-cache-dir`. If `target=/var/lib/apt` ->
## don't do `rm -rf /var/lib/apt/lists/*`.
RUN --mount=type=cache,target=/root/.cache/pip \
    apt-get -qq update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -yq \
        wget unzip libsndfile-dev \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        torch torchaudio nemo_toolkit[all] nltk torchmetrics onnx onnxruntime \
    && wget -q --content-disposition ${host}${path} -O quartznet15x5.zip \
    && unzip -q quartznet15x5.zip && mkdir -p ${asr_model_dir} \
    && wget -q ${repo}/${yaml_file} --directory-prefix quartznet15x5 \
    && python3 prepare.py quartznet15x5 && mv models /models \
    && bash ./fetch_models.sh /models \
    && rm -rf /var/lib/apt/lists/* quartznet15x5.zip

## Defaul target
FROM base
COPY --from=dev /models /models
