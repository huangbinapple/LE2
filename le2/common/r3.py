"""Some functions related to transform in 3d sapce"""
import torch
from torch import einsum
from einops import rearrange


def vec2rotation(vec):
  """(..., 3, 3) <- (..., 3, 3)
  Return the transform matrix given three points' coordinate."""
  v1 = vec[..., 2, :] - vec[..., 1, :]  # (..., 3)
  v2 = vec[..., 0, :] - vec[..., 1, :]  # (..., 3)
  e1 = v1 / vector_robust_norm(v1, dim=-1, keepdim=True)  # (..., 3)
  u2 = v2 - e1 * rearrange(einsum('...L,...L->...', e1, v2), '...->...()')
  #(..., 3)
  e2 = u2 / vector_robust_norm(u2, dim=-1, keepdim=True)
  e3 = torch.cross(e1, e2, dim=-1)  # (..., 3)
  return torch.stack((e1, e2, e3), dim=-1)  # (B, 3, 3)

def vec2transform(vec):
  """(..., 4, 4) <- (..., 3, 3)"""
  result = torch.zeros(*vec.shape[:-2], 4, 4, device=vec.device)
  result[..., :3, :3] = vec2rotation(vec)
  result[..., :3, 3] = vec[..., 1, :]
  result[..., 3, 3] = 1
  return result

def transform2feat(vec):
  """(..., 12) <- (..., 4, 4)"""
  return rearrange(vec[..., :3, :], '... X Y -> ... (X Y)')

def transform_invert(transform):
  """(..., 4, 4)"""
  result = torch.zeros(*transform.shape[:-2], 4, 4, device=transform.device)
  result[..., :3, :3] = transform[..., :3, :3].transpose(-1, -2)
  result[..., :3, -1] = einsum('...ij,...j->...i',
    -result[..., :3, :3], transform[..., :3, -1])
  # (..., 3, 3) <- (..., 3, 3), (..., 3)
  result[..., 3, 3] = 1
  return result

def vec2homo(vec):
  """(..., 4) <- (..., 3)"""
  result = torch.ones(*vec.shape[:-1], 4, device=vec.device)
  result[..., :3] = vec
  return result

def vector_robust_norm(vec, epison=1e-8, **kargs):
  """(B, 3)"""
  return torch.linalg.vector_norm(vec, **kargs) + epison
