"""Module definitions for the LE2 model."""


import torch
from torch import nn
from le2.model import feature
from le2.common import residue_constants as rc


class ResidueTypePredictor(nn.Module):
  """Predict residue type from features."""

  def __init__(self, d_input, d_model=256, n_heads=4, n_layers=3,
              device='cpu'):
    super().__init__()
    # Define the input layer.
    self.input = nn.Linear(d_input, d_model, device=device)
    # Define the transformer encoder.
    encoder_layers = nn.TransformerEncoderLayer(
      d_model, n_heads, batch_first=True, device=device)
    self.encoder = nn.TransformerEncoder(encoder_layers, n_layers)
    # Define the output layer.
    self.head_residue_type = nn.Linear(d_model, rc.restype_num + 1, device=device)

  def forward(self, sample: dict, compute_loss=False) -> dict:
    """
    Args:
      - sample: dict of input (See le2/data/dataset.py/collate_fn).
        - features: dict of features. Shape: (B, L, d_input)
        - mask: torch.Tensor of mask. Shape: (B, L)
    Return:
      - output: dict of output.
    """
    output = {}
    x = feature.make_feature(sample['features'])
    # Shape: (B, L, 45)
    x = self.input(x)  # Shape: (B, L, 256)
    x = self.encoder(x, src_key_padding_mask=~sample['mask'])
    # Shape: (B, L, 256)
    # Average over the sequence length, with mask!
    x = torch.sum(x, dim=1)  # Shape: (B, 256)
    x = x / torch.sum(sample['mask'], dim=1, keepdim=True)  # Shape: (B, 256)
    x = self.head_residue_type(x)  # Shape: (B, 21)
    output['logits'] = x
    if compute_loss:
      self.loss(output, sample)
    return output
  
  def loss(self, output, sample):
    """Compute loss."""
    output['loss'] = nn.CrossEntropyLoss()(
      output['logits'], sample['labels']['target_name'])