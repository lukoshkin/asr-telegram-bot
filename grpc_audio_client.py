#!/usr/bin/env python
# MIT License

# Copyright (c) 2022 Vladislav Lukoshkin

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import numpy as np

import ffmpeg
import grpc

import torch
import torch.nn.functional as F

import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample

from tritonclient.grpc import service_pb2, service_pb2_grpc
from tritonclient.utils import triton_to_np_dtype

# from asr_utils import CTCGreedyDecoder
# GREEDY_DECODER = CTCGreedyDecoder(" abcdefghijklmnopqrstuvwxyz'")


def read_audiofile(
        fname, squeeze_dims=True,
        mixdown=True, resample_to=None, check_audio=False):
    """
    Read audio file, ensure it comes from a distribution
    similar to that of the training dataset.
    """
    meta = torchaudio.info(fname)
    if check_audio:
        assert meta.bits_per_sample == 16, 'Only 16-bit WAV PCM supported'
        assert meta.encoding[:3] == 'PCM', 'encoding should be linear PCM'

    signal, sample_rate = torchaudio.load(fname)

    if resample_to is not None and sample_rate != resample_to:
        signal = Resample(meta.sample_rate, resample_to)(signal)
        sample_rate = resample_to

    if mixdown and len(signal) > 1 :
        signal = signal.mean(1, keepdims=True)

    if squeeze_dims:
        signal = signal.squeeze(0)

    return signal, sample_rate


class AudioTranscriber:
    """
    Simple triton-client (gRPC protocol) for audio inference.
    """
    def __init__(
            self, url, model_name, model_version,
            processor, streaming, asynchronous, batch_size):
        self.batch_size = batch_size
        self.model_name = model_name
        self.model_version = model_version
        self.processor = processor
        self.asynchronous = asynchronous
        self.streaming = streaming

        self._parse_model(url)
        self._request = self._request_template()

    def _parse_model(self, url):
        """
        Retrieve attributes necessary for the request template.
        Conduct some sanity tests.
        """
        channel = grpc.insecure_channel(url)
        self._stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

        metadata_request = service_pb2.ModelMetadataRequest(
                name=self.model_name, version=self.model_version)
        metadata_response = self._stub.ModelMetadata(metadata_request)

        config_request = service_pb2.ModelConfigRequest(
                name=self.model_name, version=self.model_version)
        config_response = self._stub.ModelConfig(config_request)

        self._input_name = metadata_response.inputs[0].name
        self._input_dtype = metadata_response.inputs[0].datatype
        self._input_shape = metadata_response.inputs[0].shape
        self._output_name = metadata_response.outputs[0].name
        self._output_dtype = metadata_response.outputs[0].datatype
        self._supports_batching = config_response.config.max_batch_size > 0
        self._sanity_checks(metadata_response, config_response.config)

    def _sanity_checks(self, model_meta, model_conf):
        """
        Check the configuration of a model to make sure it meets the
        requirements for an ASR network (as expected by this client)
        """
        input_meta = model_meta.inputs[0]
        output_meta = model_meta.outputs[0]

        if len(model_meta.inputs) != 1:
            raise Exception(
                    f'expecting 1 input, got {len(model_meta.inputs)}')
        if len(model_meta.outputs) != 1:
            raise Exception(
                    f'expecting 1 output, got {len(model_meta.outputs)}')
        if len(model_conf.input) != 1:
            raise Exception(
                    'expecting 1 input in model configuration,'
                    f' got {len(model_conf.input)}')

        if output_meta.datatype != 'FP32':
            raise Exception(
                    'expecting output datatype to be FP32, '
                    f"model '{model_meta.name}' output "
                    f'type is {output_meta.datatype}')

        ## Model's input has shape (B, N, C)
        ## ─ (batch size, #frames, alphabet size)
        expected_input_dims = [3] if self._supports_batching else [2, 3]
        if len(input_meta.shape) not in expected_input_dims:
            raise Exception(
                    'expecting input to have'
                    f"{'or'.join(map(str, expected_input_dims))}"
                    f"dimensions, model '{model_meta.name}' input "
                    f'has {len(input_meta.shape)}')

        if not self._supports_batching and self.batch_size != 1:
            raise Exception('This model does not support batching.')
        if self.asynchronous and self.streaming:
            raise Exception('Both async and streaming flags set to True')

    def _request_template(self):
        """
        Prepare a request template.
        """
        request = service_pb2.ModelInferRequest()
        request.model_name = self.model_name
        request.model_version = self.model_version

        output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
        output.name = self._output_name
        request.outputs.extend([output])

        input = service_pb2.ModelInferRequest().InferInputTensor()
        input.name = self._input_name
        input.datatype = self._input_dtype
        input.shape.extend(self._input_shape)
        request.inputs.extend([input])

        return request

    def __call__(self, audio_file, max_length_size):
        """
        Transcribe audio file.
        """
        batch_gen = self._compose_batches(audio_file, max_length_size)
        transciption = []

        if self.streaming:
            batch_gen = self._stub.ModelStreamInfer(batch_gen)

        for batch in batch_gen:
            response = self._request_batch(batch)
            transciption.append(self._postprocess(response))

        transciption = ' '.join(transciption)
        return transciption

    def _request_batch(self, batch):
        """
        Make infer request to server.
        """
        if self.streaming:
            if len(batch.error_message):
                raise Exception(batch.error_message)

            return batch.infer_response

        if self.asynchronous:
            return self._stub.ModelInfer.future(self._request).result()

        return self._stub.ModelInfer(self._request)

    def _compose_batches(self, audio, max_length_size):
        """
        Make batches out of the audio chunks. Return a generator.
        Incomplete batch is padded with a zero tensor of the feature's shape.
        """
        def _request(input_bytes):
            self._request.ClearField('raw_input_contents')
            self._request.raw_input_contents.extend([input_bytes])
            return self._request

        features = self._preprocess(audio, max_length_size)

        while len(features) >= self.batch_size:
            input_bytes = features[:self.batch_size].tobytes()
            features = features[self.batch_size:]

            if len(features) == 0:
                break

            yield _request(input_bytes)
        else:
            input_bytes = features.tobytes()
            input_bytes += np.zeros(
                (self.batch_size - len(features),
                *features[0].shape)).tobytes()

        yield _request(input_bytes)

    def _preprocess(self, signal, max_signal_length):
        segments = list(signal.split(max_signal_length, dim=-1))
        pad = max_signal_length - segments[-1].size(-1)
        pad = (pad // 2, pad - pad // 2)

        segments[-1] = F.pad(segments[-1], pad)
        segments = torch.stack(segments)

        length = torch.full((len(segments),), max_signal_length)
        segments, length = self.processor.preprocess(segments, length)

        dtype = triton_to_np_dtype(self._input_dtype)
        segments = segments.numpy().astype(dtype)

        return segments

    def _postprocess(self, response):
        """
        Convert bytes to a float tensor and decode it with a greedy decoder.
        """
        dtype = triton_to_np_dtype(self._output_dtype)
        batch = np.frombuffer(response.raw_output_contents[0], dtype)
        logits = np.reshape(batch, response.outputs[0].shape)

        if not self._supports_batching:
            logits = [logits]

        length = torch.full((len(logits),), logits[0].size(-2))
        phrase = self.processor.postprocess(logits, length)
        return phrase


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--async',
            dest='asynchronous',
            action='store_true',
            default=False,
            help='Use asynchronous inference API')
    parser.add_argument(
            '--streaming',
            action='store_true',
            default=False,
            help='Use streaming inference API')
    parser.add_argument(
            '-m',
            '--model-name',
            required=True,
            help='Name of model')
    parser.add_argument(
            '-x',
            '--model-version',
            default='',
            help='Version of model. Default is to use latest version.')
    parser.add_argument(
            '-b',
            '--batch-size',
            type=int,
            default=1,
            help='Batch size. Default is 1.')
    parser.add_argument(
            '-u',
            '--url',
            default='server:8001',
            help='Inference server URL. Default is server:8001.')
    parser.add_argument(
            'audio_file',
            nargs='?',
            help='Input audio')

    return parser


def main(args=None):
    args = vars(build_argparser().parse_args(args))

    audio_file = args.pop('audio_file')
    *new_audio_file, suffix = audio_file.split('.')

    if suffix != 'wav':
        new_audio_file = ''.join(new_audio_file) + '.wav'
        (
            ffmpeg.input(audio_file)
            .output(new_audio_file)
            .overwrite_output()
            .run()
        )
        audio_file = new_audio_file

    audio, sample_rate = read_audiofile(
            audio_file, squeeze_dims=True,
            check_audio=True, resample_to=16000)

    args['processor'] = torch.load('/workspace/processor.pt')

    # args['processor'].postprocess = GREEDY_DECODER.forward
    # args['processor'].preprocess = MelSpectrogram(
    #         sample_rate, n_fft=512, win_length=int(sample_rate*.02),
    #         n_mels=64, f_max=8000, norm='slaney', mel_scale='slaney')

    dt_max = 10
    transcriber = AudioTranscriber(**args)
    transcription = transcriber(audio, int(sample_rate*dt_max))
    print(transcription)

    return transcription.strip()


if __name__ == '__main__':
    main()
