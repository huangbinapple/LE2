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
    self.device = device
    # Define the input layer.
    self.input = nn.Linear(d_input, d_model, device=device)
    # Define the transformer encoder.
    encoder_layers = nn.TransformerEncoderLayer(
      d_model, n_heads, batch_first=True, device=device)
    self.encoder = nn.TransformerEncoder(encoder_layers, n_layers)
    # Define the output layer.
    self.head_residue_type = nn.Linear(d_model, rc.restype_num + 1, device=device)

  def forward(self, sample: dict, output_loss=False,
              output_logit=False, output_confidence=False,
              output_predicted_rtype=False, output_ncorrect=False) -> dict:
    """
    Args:
      - sample: dict of input (See le2/data/dataset.py/collate_fn).
        - features: dict of features. Shape: (B, L, d_input)
        - mask: torch.Tensor of mask. Shape: (B, L)
    Return:
      - output: dict of output.
        - logits: torch.Tensor of logits. Shape: (B, 21)
        - loss (if compute_loss): torch.Tensor of loss. Shape: ()
    """
    mask = sample['mask'].to(self.device)
    output = {}
    x = feature.make_feature(sample['features'], device=self.device)
    # Shape: (B, L, 45)
    x = self.input(x)  # Shape: (B, L, 256)
    x = self.encoder(x, src_key_padding_mask=~mask) # Shape: (B, L, 256)
    # Average over the sequence length, with mask!
    x = torch.sum(x, dim=1)  # Shape: (B, 256)
    x = x / torch.sum(mask, dim=1, keepdim=True) # Shape: (B, 256)
    logits = self.head_residue_type(x)  # Shape: (B, 21)
    
    # Calculation only depends on the logits.
    if output_logit:
      output['logit'] = logits
    if output_predicted_rtype or output_ncorrect:
      predicted_rtypes = torch.argmax(logits, dim=1)  # Shape: (B,)
      if output_predicted_rtype:
        output['predicted_rtype'] = predicted_rtypes
    if output_confidence or output_confidence:
      probabilties = torch.softmax(logits, dim=1)  # Shape: (B, 21)
      if output_confidence:
        output['probability'] = probabilties
      if output_confidence:
        # Calculate the confidence, which is the entropy of the probability
        confidence = (probabilties * torch.log(probabilties)).sum(1) # Shape: (B,)
        output['confidence'] = confidence
    
    # Calculation depends on the target.
    if output_loss or output_ncorrect:
      target_names = sample['labels']['target_name'].to(self.device)
      if output_loss:
        output['loss'] = nn.CrossEntropyLoss(reduction='none')(
        logits, target_names.to(self.device))      
      if output_ncorrect:
        output['ncorrect'] = (predicted_rtypes == target_names).sum()
      
    return output