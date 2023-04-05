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