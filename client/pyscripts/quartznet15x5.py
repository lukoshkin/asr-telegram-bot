#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.onnx import export as to_onnx

from nemo.collections.asr.models import EncDecCTCModel
from omegaconf import OmegaConf


def parse_args():
    """Returns parsed command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default='model.onnx',
            help='path where to save the ONNX model')
    parser.add_argument('--model-name', default='QuartzNet15x5Base-En',
            help='name of the pretrained model.'
            " Alternative to the 'components' positional")
    parser.add_argument('-s', '--inp-shape', default='(1, 64, 1024)',
            type=lambda s: eval(s), help='a stringified list of'
            " the model's input dimensions")
    parser.add_argument('components', nargs='?',
            help='folder with encoder-decoder weights'
            ' and the yaml file of the model config')

    args = parser.parse_args()
    return args


def load_qn_model(components: str=None, model_name: str=None):
    """Loads Quartznet15x5 model either from a directory with
    downloaded components or from NeMo hub of models by its name.
    """
    ## model 'components' have higher precedence than 'model_name'
    if components is None and model_name is not None:
        ref_model = EncDecCTCModel.from_pretrained(model_name=model_name)
        return ref_model

    components = Path(components)
    assert components.exists(), 'Folder with model components not found'
    try:
        enc_path = next(components.glob('JasperEncoder*.pt'))
        dec_path = next(components.glob('JasperDecoder*.pt'))
        qn_yaml = next(components.glob('quartznet*.yaml'))
    except StopIteration as exc:
        raise Exception(
                'Missing some of the following: model config (yaml), '
                'encoder weights (JasperEncoder*.pt), decoder weights'
                ' (JasperDecoder*.pt)') from exc

    map_location = {'map_location': torch.device('cpu')}
    enc_state = torch.load(enc_path, **map_location)
    dec_state = torch.load(dec_path, **map_location)

    ## obsolete NOTE: different names of yaml file depending on the source.
    ## https://catalog.ngc.nvidia.com/orgs/nvidia/models      ->  quartznet15x5.yaml
    ## https://github.com/NVIDIA/NeMo/main/examples/asr/conf  ->  quartznet_15x5.yaml
    ## Currently, the git version of the yaml file is more appropriate.
    cfg = OmegaConf.load(qn_yaml).model
    cfg.pop('validation_ds')
    cfg.pop('train_ds')

    ref_model = EncDecCTCModel(cfg=cfg)
    ref_model.encoder.load_state_dict(enc_state)
    ref_model.decoder.load_state_dict(dec_state)

    ## These prerpocessing parameters are set to zero in
    ## `EncDecCTCModel.transcribe` method, however, we don't need them here,
    ## since we convert the model to onnx w/o the preprocessor component.
    # ref_model.preprocessor.featurizer.dither = 0.
    # ref_model.preprocessor.featurizer.pad_to = 0

    return ref_model


## The preprocessing step in EncDecCTCModel model uses
## unsupported at the moment TorchScript operator: `aten::stft`.
## Therefore, we pass (processed_signal, processed_signal_length) kwargs.
class ComputingModel(nn.Module):
    """
    Currently, we consider only inputs of unit batch dimension.
    Therefore, input length can be easily inferred.
    """
    def __init__(self, ref_model):
        super().__init__()
        self.core = ref_model

    def forward(self, signal):
        length = torch.full((1,), signal.size(-1))
        logits, *_ = self.core.forward(
                processed_signal=signal,
                processed_signal_length=length)
        return logits  # return only logits
## See https://pytorch.org/docs/stable/onnx_supported_aten_ops.html
## if the status of `aten::stft` has changed.


def main():
    """Moved to a separte function just for better readability."""
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ref_model = load_qn_model(args.components, args.model_name)

    ## Prepare model for evaluation.
    ref_model.eval()
    ref_model.encoder.freeze()
    ref_model.decoder.freeze()

    model = ComputingModel(ref_model)
    signal = torch.empty(*args.inp_shape)
    ## signal.size(-1) = 1024 is 10.16s at sampling rate 16kHz and other
    ## default parameters in the preprocessing step of the quartznet model.
    # length = torch.full((len(signal),), signal.size(-1))

    # to_onnx(model, (signal, length),
    to_onnx(model, signal, str(output),  # won't work with Path object
            input_names=['processed_signal'], output_names=['logits'])
            # input_names=('inp', 'inp_len'), output_names=('out', 'out_len'))


## Though it won't be used in any other way..
if __name__ == '__main__':
    main()
