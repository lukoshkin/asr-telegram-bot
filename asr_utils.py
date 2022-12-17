import logging
from typing import Generator, Iterable

import torch
import torch.nn.functional as F
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram


def preemphasized(signal: torch.Tensor, alpha=.97) -> torch.Tensor:
    """Improves signal-to-noise ratio.
    (Maybe redundant here since it is from older system designs.)
    """
    return torch.cat((signal[:1], signal[1:] - alpha * signal[:-1]))


def normalized(tensor: torch.Tensor) -> torch.Tensor:
    """Mean-variance normalization of the input tensor."""
    mu = tensor.mean(-1, keepdim=True)
    sigma = tensor.std(-1, keepdim=True)
    tensor = (tensor - mu) / (sigma + 1e-5)

    return tensor


class AudioPreprocessor(MelSpectrogram):
    """Local to repo audio preprocessor.
    Obtains log scaled mel-spectrogram normalized per feature.
    """
    def __init__(self, sample_rate: int,
                 log_zero_guard_value=2**-24, **kwargs):
        self.log0_guard_val = log_zero_guard_value
        in_frames = lambda t: int(t * sample_rate) if t < 1 else t

        if 'win_length' in kwargs:
            assert kwargs['win_length'] > 0, "'win_length' can't be negative"
            kwargs['win_length'] = in_frames(kwargs['win_length'])

        if 'hop_length' in kwargs:
            kwargs['hop_length'] = in_frames(kwargs['hop_length'])

        super().__init__(sample_rate, **kwargs)
        ## NOTE: Setting `kwargs['normalized'] = True` is not
        ## the same as normalizing in the forward method

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        signal = super().forward(preemphasized(signal))
        signal = normalized((signal + self.log0_guard_val).log())
        return signal


class CTCDecoder(torch.nn.Module):
    def __init__(self, labels: str | list, blank: str='-', silence: str='|'):
        super().__init__()
        assert isinstance(labels, Iterable) and isinstance(labels[0], str)
        assert len(set(labels)) == len(labels), "Repeating chars in 'labels'"
        assert isinstance(blank, str) and isinstance(silence, str)

        self.labels = list(labels) + [blank, silence]
        self.blank_id = self.labels.index(blank)

    def __call__(self, raw_tokens: torch.Tensor) -> list[list[str]]:
        tokens, lens = self.process(raw_tokens)

        texts = []
        for r, l in zip(tokens, lens):
            text = ''.join([self.labels[i] for i in r[:l]])
            text = text.replace('|', ' ').strip().split()
            texts.append(text)

        return texts

    def process(self, mat: torch.Tensor) -> torch.Tensor:
        assert isinstance(mat, torch.Tensor), 'Expects torch.Tensor as input'
        assert not torch.is_floating_point(mat), 'Expects input of int dtype'

        ## Locate original entries in the unique consecutive array (1D)
        ## [[13, 7, 7, 21, 9],   ->    [[0, 1, 1, 2, 3],
        ##  [0,  0, 9,  4, 2]]          [4, 4, 5, 6, 7]]
        ids = torch.unique_consecutive(mat, return_inverse=True)[1]
        ## Zero the starting index of each row
        ids -= ids.min(dim=1, keepdims=True)[0]

        ## Fill 'blank' matrix of the same size with `mat`'s data placing
        ## consecutively repeated elements of each row under the same index.
        ## cons_uniq[i][ids[i][j]] = mat[i][j]
        cons_uniq = torch.full_like(mat, self.blank_id)
        cons_uniq = cons_uniq.scatter_(1, ids, mat)

        lens = (cons_uniq != self.blank_id).sum(1)
        maxlen = lens.max()

        tokens = mat.new(len(lens), maxlen)
        tokens[:] = torch.arange(maxlen)
        tokens -= lens[:, None]
        mask = tokens < 0

        tokens[mask] = cons_uniq[cons_uniq != self.blank_id]
        tokens[~mask] = self.blank_id
        return tokens, lens


## The next two function are in a beta state of development.
## They may serve as a more sensible splitting of a signal in chunks.
def split_on_silence(
        signal: torch.Tensor, frame_size: int=2048,
        hop_size: int=512, top_db: int=None) -> torch.Tensor:
    """
    Pytorch implementation of `librosa.effects.split`
    limited to 1-channel input signals.
    """
    assert signal.ndim == 1 or signal.shape[0] == 1

    n_samples = signal.size(-1)
    n_bins = n_samples // frame_size
    pad = (n_bins + 1) * frame_size - n_samples

    if pad > 0 and pad < frame_size:
        signal = F.pad(signal, (0, pad))

    signal = signal.unfold(-1, frame_size, hop_size)  # along -1 axis
    signal = signal.square().mean(-1).sqrt()  # RMS
    ## Default choice of parameters is not always OK. You can think of
    ## improving the next code line, adding more function arguments
    ## (top_db may also be involved).
    signal = AmplitudeToDB('ampl')(signal.flatten())

    assert top_db is None or top_db > 0
    threshold = signal.mean() if top_db is None else -top_db

    non_silent = signal > threshold
    edges = torch.diff(non_silent).nonzero().view(-1)
    edges += 1  # Adding to an empty tensor is also OK

    cat_list = [edges.new_tensor(1)] if non_silent[0] else []
    cat_list.append(edges)

    if non_silent[-1]:
        cat_list.append(edges.new_tensor([len(non_silent)]))

    edges = torch.cat(cat_list)

    if len(edges):
        edges *= hop_size
        edges[-1] = min(edges[-1].item(), n_samples)
        return edges.view(-1, 2)

    logging.warning(" Too high or too low 'top_db' value!")
    return edges


def on_silence_generator(
        signal: torch.Tensor, box_size: int,
        air_bubble: float=.05) -> Generator[torch.Tensor, None, None]:
    """Splits signals in chunks of size `box_size`.
    `air_bubble` - minimal space between signal chunks.
    """
    assert air_bubble > 0 and 2*air_bubble < 1
    air = int(air_bubble * box_size)

    left_space = box_size
    stack_meta, stack = [0], [[-1, 0]]
    ## -1 or None ─ doen't matter, we won't use it

    cuts = split_on_silence(signal).tolist()
    cuts.append(None)
    cuts = cuts[::-1]
    cut = cuts.pop()

    while cut is not None:
        net = cut[1] - cut[0]
        gross = net + 2*air

        if gross <= left_space:
            left_space -= gross
            ## Mind the order of the next two lines
            stack.append(cut)
            cut = cuts.pop()
            continue

        if len(stack) - stack_meta[-1] == 0 or (gross > box_size
                and gross + 2*air - left_space <= box_size):
            ## If the current box is empty, put there as much as it fits.
            ## If it is not, then check that it will be possible to pack the
            ## chopped off part in a new empty box, but not the whole cargo.

            ## Mind the order of the next two lines
            stack.append([cut[0], cut[0] + left_space - 2*air])
            cut = (cut[0] + left_space - 2*air, cut[1])

        ## Prepare a new empty box
        left_space = box_size
        stack_meta.append(len(stack)-1)

    chunk = torch.Tensor([])
    ## Mind the order of the next two lines
    stack_meta.append(len(stack)-1)
    stack.append([signal.size(1), -1])
    ## -1 or None ─ doen't matter, we won't use it

    i = 1
    box = []
    box_count = stack_meta[1] - stack_meta[0]

    for (_, a), (b, c), (d,_) in zip(stack, stack[1:], stack[2:]):
        ids = slice(max((a+b)//2, b-air), min(c+air, (c+d)//2))
        box.append(signal[..., ids])

        if len(box) >= box_count:
            chunk = torch.cat(box, dim=-1)

            box = []
            i = min(i + 1, len(stack_meta)-1)
            box_count = stack_meta[i] - stack_meta[i-1]
            yield chunk
