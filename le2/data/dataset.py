import logging
import os
import torch
from torch.utils.data import Dataset
from le2.common.protein import Protein
from le2.common import residue_constants as rc
from le2 import config
from torch.nn.utils.rnn import pad_sequence


class LocalEnvironmentDataSet(Dataset):
  def __init__(self, file_path: str, radius: float =12.0):
    logging.info(f"Loading data from {file_path}..., radius={radius}")
    self.radius = radius
    self.file_path = file_path
    file_type = file_path.split('.')[-1]
    with open(file_path) as f:
      self.protien = Protein(f.read(), file_type)
    
  def __len__(self) -> int:
    return len(self.protien)
  
  def __getitem__(self, index: int) -> dict:
    """
    Output (dict): containing the features, label, and meta info of one sample
    - feature (dict): a dictionary containing the features of the target residue
      at index `index`
      - neighbor_names (list): a list of the names of the residues that are
        in the local environment of the residue at index `index`.
      - neighbor_indicies (list): a list of the indicies of the residues
      - neighbor_chain_ids (list): a list of the chain ids of the residues
      - neighbor_atom_cooridnates (tensor): a tensor of the atom coordinates
      - target_index (int): the index of the target residue
      - target_chain_id (str): the chain id of the target residue
      - target_atom_coordinates (tensor): a tensor of the atom coordinates
    - label (dict):
      - target_name (str): the name of the residue at index `index`
    - meta (dict): a dictionary containing the metadata of the residue at
        index `index`
      - file_path (str): the path to the file that contains the protein
    """
    features = {}
    neighbor_indicies = self.protien.get_neighbor_indicies(index, self.radius)
    features['neighbor_names'] =\
      [self.protien.residue_names[i] for i in neighbor_indicies]
    features['neighbor_indicies'] =\
      [self.protien.residue_indicies[i] for i in neighbor_indicies]
    features['neighbor_chain_ids'] =\
      [self.protien.residue_chain_ids[i] for i in neighbor_indicies]
    features['neighbor_atom_coordinates'] =\
      self.protien.atom_coords[neighbor_indicies]
    features['target_index'] = self.protien.residue_indicies[index]
    features['target_chain_id'] = self.protien.residue_chain_ids[index]
    features['target_atom_coordinates'] = self.protien.atom_coords[index]
    
    label = dict(target_name=self.protien.residue_names[index])
    
    meta = dict(file_path=self.file_path)
    return {'feature': features, 'label': label, 'meta': meta}


def collate_fn(batch: list) -> dict:
  """
  Collate function for the LocalEnvironmentDataSet.
  
  Args:
    batch (list): a list of samples from the LocalEnvironmentDataSet.
    
  Returns (dict): a dictionary containing the features, labels, and metas of
    the batch.
    - features (dict): a dictionary containing the features of the batch.
      - neighbor_names (torch.tensor): shape: (batch_size, max_len)
      - neighbor_indicies (torch.tensor): shape: (batch_size, max_len)
      - neighbor_chain_ids (torch.tensor): shape: (batch_size, max_len)
      - neighbor_atom_coordinates (torch.tensor):
        shape: (batch_size, max_len, 3, 3)
      - target_index (torch.tensor): shape: (batch_size)
      - target_chain_id (torch.tensor): shape: (batch_size)
      - target_atom_coordinates (torch.tensor): shape: (batch_size, 3, 3)
    - labels (dict): a dictionary containing the labels of the batch.
     - target_name (torch.tensor): shape: (batch_size)
    - metas (dict): a dictionary containing the metas of the batch.
      - file_path (list): a list of the file paths of the proteins in the batch.
    - mask (torch.tensor): shape: (batch_size, max_len)
  """
  batch_size = len(batch)
  lengths = [len(sample['feature']['neighbor_names']) for sample in batch]
  max_len = max(lengths)
  output = dict(features={}, labels={}, meta={}, mask=None)
  
  # Make mask
  output['mask'] = torch.zeros(batch_size, max_len, dtype=torch.bool)
  for i, length in enumerate(lengths):
    output['mask'][i, :length] = True
    
  output['features']['neighbor_names'] = pad_sequence(
    [torch.tensor(list(
      map(rc.get_resname_index, sample['feature']['neighbor_names'])))
      for sample in batch],
    batch_first=True)
  
  output['features']['neighbor_indicies'] = pad_sequence(
    [torch.tensor(sample['feature']['neighbor_indicies'])
     for sample in batch], batch_first=True)
  
  output['features']['neighbor_chain_ids'] = pad_sequence(
    [torch.tensor(list(map(hash, sample['feature']['neighbor_chain_ids'])))
     for sample in batch], batch_first=True)
  
  output['features']['neighbor_atom_coordinates'] = pad_sequence(
    [sample['feature']['neighbor_atom_coordinates']
     for sample in batch], batch_first=True)
  
  output['features']['target_index'] = torch.tensor(
    [sample['feature']['target_index'] for sample in batch])
  
  output ['features']['target_chain_id'] = torch.tensor(
    list(map(hash, [sample['feature']['target_chain_id'] for sample in batch])))

  output['features']['target_atom_coordinates'] = torch.stack(
    [sample['feature']['target_atom_coordinates'] for sample in batch])
  
  output['labels']['target_name'] = torch.tensor(
    [rc.get_resname_index(sample['label']['target_name']) for sample in batch])
    
  output['meta']['file_path'] = [sample['meta']['file_path'] for sample in batch]
  
  return output