from typing import Iterator, Sequence
from torch.utils.data.sampler import Sampler
import torch


def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def convert_to_sparse_coo(tensor, cutoff):
  # Create a mask tensor that's 1 where the input tensor is below the cutoff,
  # and 0 where it's above the cutoff.
  mask = tensor < cutoff
  
  # Get the indices of the non-zero elements in the mask tensor.
  indices = torch.nonzero(mask, as_tuple=False)
  
  # Create a new sparse COO tensor using the non-zero elements of the input
  # tensor and their corresponding indices.
  values = tensor[mask]
  sparse_tensor = torch.sparse_coo_tensor(indices.t(), values, size=tensor.shape)
  
  return sparse_tensor


class FixedOrderSampler(Sampler[int]):
  r"""Samples elements from a given list of indices in a fixed order,
    without replacement. A modified version of SubsetRandomSampler.

  Args:
    indices (sequence): a sequence of indices
  """
  indices: Sequence[int]

  def __init__(self, indices: Sequence[int]) -> None:
    self.indices = indices

  def __iter__(self) -> Iterator[int]:
    for i in range(len(self.indices)):
      yield self.indices[i]

  def __len__(self) -> int:
    return len(self.indices)
