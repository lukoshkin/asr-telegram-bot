import os
import sys
from fastapi import FastAPI
import grpc_image_client as client

app = FastAPI()


@app.get("/")
def load_voice_mute():
    cnum = int(os.environ.get('CLASSIFIER_TOPK', '1'))
    model = os.environ.get('CLASSIFIER_MODEL', 'inception_graphdef')

    arg_str = f'-m {model} -c {cnum} -s INCEPTION input_image.jpeg'
    sys.argv[1:] = arg_str.split()
    msg = client.main()

    return {"message": msg}
