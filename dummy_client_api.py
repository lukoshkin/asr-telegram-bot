import os
import sys
from fastapi import FastAPI

import grpc_image_client
import grpc_audio_client

app = FastAPI()


@app.get("/image")
def classify_image():
    cnum = int(os.environ.get('CLASSIFIER_TOPK', '1'))
    model = os.environ.get('CLASSIFIER_MODEL', 'inception_graphdef')

    arg_str = f'-m {model} -c {cnum} -s INCEPTION input_image.jpeg'
    sys.argv[1:] = arg_str.split()
    msg = grpc_image_client.main()

    return {"message": msg}

@app.get("/voice")
def transcribe_voice():
    model = os.environ.get('ASR_MODEL', 'quartznet15x5')

    arg_str = f'--async -m {model} user_voice.ogg'
    sys.argv[1:] = arg_str.split()
    msg = grpc_audio_client.main()

    return {"message": msg}
