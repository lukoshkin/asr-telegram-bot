import torch

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
