FROM ubuntu:20.04

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update \
    && apt-get install -yq \
        ffmpeg python3 python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade \
    "uvicorn[standard]" fastapi Pillow ffmpeg-python \
    tritonclient[grpc] torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cpu

WORKDIR /workspace
ENTRYPOINT [ "uvicorn", "dummy_client_api:app", "--host", "0.0.0.0", "--port", "80" ]
