import logging

import torch
import torch.nn.functional as F
from torchaudio.transforms import AmplitudeToDB


def preemphasized(signal: torch.Tensor, alpha=.97) -> torch.Tensor:
    """
    Improves signal-to-noise ratio.
    (Maybe redundant here since it is from older system designs.)
    """
    return torch.cat((signal[:1], signal[1:] - alpha * signal[:-1]))


def normalized(tensor: torch.Tensor) -> torch.Tensor:
    """
    Mean-variance normalization of the input tensor.
    """
    mu = tensor.mean(-1, keepdim=True)
    sigma = tensor.std(-1, keepdim=True)
    tensor = (tensor - mu) / (sigma + 1e-5)

    return tensor


def split_on_silence(signal, frame_size=2048, hop_size=512, top_db=None):
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


def on_silence_generator(signal, box_size, air_bubble=.05):
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


## Original code:
# https://pytorch.org/audio/main/tutorials/asr_inference_with_ctc_decoder_tutorial.html
class CTCGreedyDecoder(torch.nn.Module):
    def __init__(self, labels, blank=-1, blank_not_in_labels=True):
        super().__init__()
        self.labels = list(labels)
        self.blank = blank % (len(self.labels) + blank_not_in_labels)

    def forward(self, logits):
        """
        Choose the most probably letter at each step.

        Args:
          logits (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        logits = torch.Tensor(logits.copy()[0])
        indices = torch.argmax(logits, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([self.labels[i] for i in indices])

        return joined.replace("|", " ").strip().split()
