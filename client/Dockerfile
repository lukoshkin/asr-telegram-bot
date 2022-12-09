# syntax = docker/dockerfile:1.3
## ↑ Required is using Docker's BuildKit ↑
FROM ubuntu:20.04

## BuildKit allows setting `--mount` option of `RUN` instruction.
## Mounts don't help (speed up building) if the layer is cached. But here,
## they are given as a usage example. Note: you shouldn't delete what you are
## going to mount to the host. E.g., if `target=/root/.cache/pip`, then
## don't specify the option `--no-cache-dir`. If `target=/var/lib/apt` ->
## don't do `rm -rf /var/lib/apt/lists/*`.
RUN --mount=type=cache,target=/root/.cache/pip \
    apt-get -qq update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -yq \
        ffmpeg python3 python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio tritonclient[grpc] \
    uvicorn[standard] fastapi Pillow ffmpeg-python

WORKDIR /workspace
ENTRYPOINT [ "uvicorn", "dummy_client_api:app", \
    "--host", "0.0.0.0", "--port", "80" ]
