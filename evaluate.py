#!/usr/bin/env python3
"""
This project is not aimed to be efficient; thus,
we use python functions and classes to calculate metrics.
"""
import json
import argparse
import logging

import torch
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram
from torchaudio.datasets import LIBRISPEECH
from nltk.metrics.distance import edit_distance

from grpc_audio_client import AudioTranscriber, build_argparser, read_audiofile
from asr_utils import split_on_silence


def normalized_WED(pred, true):
    if len(true) == 0 or len(pred) == 0:
        return 0

    denom = len(true) if len(true) > len(pred) else len(pred)
    norm_ED = 1 - edit_distance(pred, true) / denom
    return norm_ED


def direct_pred(transcriber, signal, frame_size):
    pred = transcriber(signal, frame_size)
    pred = pred.stip().upper()
    return pred


def split_collect_pred(transcriber, signal, frame_size):
    preds = []
    for chunk in on_silence_generator(signal, frame_size):
        pred = transcriber(chunk, frame_size)
        pred = pred.upper()

    return ''.join(preds)


def main(prediction_scheme, print_to_stdout):
    if not print_to_stdout:
        print('Silent mode: ON')

    args = vars(build_argparser().parse_args(['-m', 'quartznet15x5']))
    args.pop('audio_file')

    dt = 1.27
    sample_rate = 16000
    args['transform'] = MelSpectrogram(
            sample_rate, n_fft=512, win_length=int(sample_rate*.02),
            n_mels=64, f_max=8000, norm='slaney', mel_scale='slaney')

    librispeech = LIBRISPEECH('./data', url='test-clean', download=True)
    data_loader = DataLoader(librispeech, batch_size=1, shuffle=True)
    transcriber = AudioTranscriber(**args)

    scores = {'NWED': 0, 'Accuracy': 0}
    denom = 0

    for signal, sr, (true,), *_ in data_loader:
        if sr != sample_rate:
            logging.warn(f'Inappropriate sample rate: {sr}')
            continue

        pred = prediction_scheme(transcriber, signal, int(sample_rate * dt))

        denom += 1
        scores['NWED'] += normalized_WED(pred, true)
        scores['Accuracy'] += pred == true

        if print_to_stdout:
        # if print_to_stdout and len(pred.strip()):
            print(f'<{pred}>', f'<{true}>', sep='\n-----\n')
            print('\n\n')

    scores = {k: v/denom for k,v in scores.items()}
    with open('scores.json', 'w') as fp:
        json.dump(scores, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--print',
            action='store_true',
            default=False,
            help='Print prediction and true text to STDOUT')
    parser.add_argument(
            '--scheme',
            default='direct',
            help='Print prediction and true text to STDOUT')
    args = parser.parse_args()
    scheme = direct_pred if args.scheme == 'direct' else split_collect_pred
    main(scheme, args.print)
