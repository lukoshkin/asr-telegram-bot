import argparse
from pathlib import Path

import torch
from torch import nn
from torch.onnx import export as to_onnx

from nemo.collections.asr.models import EncDecCTCModel
from omegaconf import OmegaConf


parser = argparse.ArgumentParser()
parser.add_argument(
        '--output-dir',
        default='.',
        help='folder to save the ONNX model')
parser.add_argument(
        'model_components',
        help='folder with encoder-decoder weights'
        ' and the yaml file of the model config')

args = parser.parse_args()
output_dir = Path(args.output_dir)
components_folder = Path(args.model_components)

enc_path = components_folder / 'JasperEncoder-STEP-247400.pt'
dec_path = components_folder / 'JasperDecoderForCTC-STEP-247400.pt'
map_location = {'map_location': torch.device('cpu')}
enc_state = torch.load(enc_path, **map_location)
dec_state = torch.load(dec_path, **map_location)

## NOTE: different names of yaml file depending on the source.
## https://catalog.ngc.nvidia.com/orgs/nvidia/models -> quartznet15x5.yaml
## https://github.com/NVIDIA/NeMo/main/examples/asr/conf -> quartznet_15x5.yaml
## Currently, the git version of the yaml file is more appropriate.
cfg = OmegaConf.load(components_folder / 'quartznet_15x5.yaml').model
cfg.pop('validation_ds')
cfg.pop('train_ds')

## Prepare model for evaluation.
ref_model = EncDecCTCModel(cfg=cfg)
ref_model.encoder.load_state_dict(enc_state)
ref_model.decoder.load_state_dict(dec_state)
ref_model.preprocessor.featurizer.dither = 0.
ref_model.preprocessor.featurizer.pad_to = 0

ref_model.eval()
ref_model.encoder.freeze()
ref_model.decoder.freeze()

## The preprocessing step in EncDecCTCModel model uses
## currently unsupported TorchScript operator: `aten::stft`.
## Therefore, we pass (processed_signal, processed_signal_length) kwargs.
class ComputingModel(nn.Module):
    def forward(self, processed_signal, processed_signal_length):
        return ref_model.forward(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length)
## See https://pytorch.org/docs/stable/onnx_supported_aten_ops.html
## if the status of `aten::stft` has changed.

## One more wrapper.
## By saving it as a torch model we can get rid of bulky nemo library
## and its dependencies in the 'base' docker image.
class ProcessingModel(nn.Module):
    def preprocess(self, input_signal, length):
        """Returns processed_signal, processed_signal_length"""
        return ref_model.preprocessor(
                input_signal=input_signal, length=length)

    def postprocess(self, logits, logits_len):
        """Returns transcription"""
        return ref_model.decoding.ctc_decoder_predictions_tensor(
                logits, logits_len)[0]


processor = ProcessingModel()
torch.save(processor, output_dir / 'processor.pt')

model = ComputingModel()
signal = torch.rand(1, 64, 1024)
# signal.size(-1) = 1024 is 10.16s at sampling rate 16kHz and other
# default parameters in the preprocessing step of the quartznet model.
length = torch.full((len(signal),), signal.size(-1))
to_onnx(model, (signal, length),
        f'{output_dir}/model.onnx',  # expects str, not Path
        input_names=('inp', 'inp_len'))


## Some tests.
# import torchaudio

# import onnx
# import onnxruntime as ort

# signal, sr = torchaudio.load('example2.wav')
# length = torch.full((len(signal),), signal.size(-1))

# signal, length = processor.preprocess(signal, length)
# to_onnx(model, (signal, length), 'dummy.onnx', input_names=('inp', 'inp_len'))

# logits, logits_len, _ = model(signal, length)
# transcription = processor.postprocess(logits, logits_len)
# assert transcription == ref_model.transcribe(['example2.wav'])
## assertion: [just_one_string1] = [just_one_string2]

# onnx_model = onnx.load("dummy.onnx")
# onnx.checker.check_model(onnx_model)

# session = ort.InferenceSession('dummy.onnx')
# _ = session.run(None, {'inp': signal.numpy(), 'inp_len': length.numpy()})
