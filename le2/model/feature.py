import torch
from le2.common import r3
from le2.common import residue_constants as rc
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def make_feature(feature_batch: dict, device: str ='cpu',
                 max_1d_distance: int = 5) -> torch.Tensor:
  """
  Make features that fed into the model.
  """
  # relative position in sequence (one hot encode)
  target_position_1d = \
    feature_batch['target_index'] + feature_batch['target_chain_id']
  neighbor_position_1d = \
    feature_batch['neighbor_indicies'] + feature_batch['neighbor_chain_ids']
  r_position_1d = neighbor_position_1d - target_position_1d.unsqueeze(1)
  r_position_1d = torch.clamp(r_position_1d + max_1d_distance,
                              min=0, max=max_1d_distance * 2)
  r_position_1d = F.one_hot(r_position_1d, num_classes=max_1d_distance * 2 + 1)
  # Shape: (B, L, 11)  # default max_1d_distance = 5
  
  # relative position in 3d space
  target_frame = r3.vec2transform(feature_batch['target_atom_coordinates'])
  # Shape: (B, 4, 4)
  neighbor_frame = r3.vec2transform(feature_batch['neighbor_atom_coordinates'])
  # Shape: (B, L, 4, 4)
  r_position_3d = torch.einsum('Bij,BLjk->BLik',
    r3.transform_invert(target_frame), neighbor_frame)
  # Shape: (B, L, 4, 4)
  r_position_3d = r3.transform2feat(r_position_3d)
  # Shape: (B, L, 12)
  
  # residue type (one hot encode)
  residue_type = F.one_hot(
    feature_batch['neighbor_names'], num_classes=rc.restype_num + 1)
  # Shape: (B, L, 21)
  
  # is on the same chain
  is_same_chain = feature_batch['neighbor_chain_ids'] - \
    feature_batch['target_chain_id'].unsqueeze(1)
  # Shape: (B, L)
  is_same_chain = is_same_chain.unsqueeze(-1)
  # Shape: (B, L, 1)
  
  # Concate all features.
  return torch.cat([
    r_position_1d,
    r_position_3d,
    residue_type,
    is_same_chain,], dim=-1)
  # Shape: (B, L, 45) 11 + 12 + 21 + 1
  