# syntax = docker/dockerfile:1.3
## ↑ Required is using Docker's BuildKit ↑
FROM python:3.10.9-slim as base
## Generally ubuntu:20.04 is OK, but I prefer using typing features
## that are only available in versions 3.10 and higher.

## Better to specify pip dependencies in a requirement.txt file
## This will be fixed in the next patches.
RUN apt-get -qq update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -yq ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir \
        uvicorn[standard] fastapi Pillow ffmpeg-python \
        tritonclient[grpc] torch torchvision torchaudio \
        --extra-index-url https://download.pytorch.org/whl/cpu

WORKDIR /workspace
ENTRYPOINT [ "uvicorn", "dummy_client_api:app", \
    "--host", "0.0.0.0", "--port", "80" ]

FROM base as dev
COPY . ./

## BuildKit allows setting `--mount` option of `RUN` instruction.
## Mounts don't help (speed up building) if the layer is cached. They are
## particularly useful when frequently rebuilding an image in search of the
## best docker recipe. Here 'mount` option is given as a usage example.
## Note: you shouldn't delete what you are going to mount to the host. E.g.,
## if `target=/root/.cache/pip`, then don't specify `--no-cache-dir` to pip.
## If `target=/var/lib/apt` → don't do `rm -rf /var/lib/apt/lists/*` after apt.
RUN --mount=type=cache,target=/root/.cache/pip \
    apt-get -qq update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -yq build-essential wget unzip libsndfile-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install cython onnxruntime nemo_toolkit[all] \
    && cp -R configs /models \
    && ./prepare_models.sh /models
## 'onnxruntime' is for testing model conversion to ONNX format,
## the inference quality before launching 'tritonserver'.

FROM base
COPY --from=dev /models /models
