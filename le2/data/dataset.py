import logging
import time
import os
import torch
from torch.utils.data import Dataset, ConcatDataset
from le2.common.protein import Protein
from le2.common import residue_constants as rc
from le2 import config
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(config.LOG_NAME)


class LocalEnvironmentDataSet(Dataset):
  def __init__(self, file_path: str, radius: float =12.0, cache_dir: str =''):
    logger.info(f"Loading data from {file_path}..., radius={radius}")
    self.radius = radius
    self.file_path = file_path
    file_type = file_path.split('.')[-1]
    if cache_dir is not None:
      if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
      cache_file = os.path.join(cache_dir, os.path.basename(file_path) + '.pt')
      if os.path.exists(cache_file):
        logger.info(f"Loading cached data from {cache_file}")
        self.protein = torch.load(cache_file)
        return
      else:
        with open(file_path) as f:
          self.protein = Protein(f.read(), file_type)
        logger.info(f"Saving data to {cache_file}")
        torch.save(self.protein, cache_file)
    logger.info(f"Loaded {len(self.protein)} residues from {file_path}")
    
  def __len__(self) -> int:
    return len(self.protein)
  
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
    neighbor_indicies = self.protein.get_neighbor_indicies(index, self.radius)
    features['neighbor_names'] =\
      [self.protein.residue_names[i] for i in neighbor_indicies]
    features['neighbor_indicies'] =\
      [self.protein.residue_indicies[i] for i in neighbor_indicies]
    features['neighbor_chain_ids'] =\
      [self.protein.residue_chain_ids[i] for i in neighbor_indicies]
    features['neighbor_atom_coordinates'] =\
      self.protein.atom_coords[neighbor_indicies]
    features['target_index'] = self.protein.residue_indicies[index]
    features['target_chain_id'] = self.protein.residue_chain_ids[index]
    features['target_atom_coordinates'] = self.protein.atom_coords[index]
    
    label = dict(target_name=self.protein.residue_names[index])
    
    meta = dict(file_path=self.file_path)
    return {'feature': features, 'label': label, 'meta': meta}
  
  
def construct_dataset_from_dir(dir_path: str, cache_dir: str='') -> ConcatDataset:
  """
  Construct a ConcatDataset from a directory using all cif and pdb
  files in that dir, by constructing a LocalEnvironmentDataSet for each
  of those files, and then Concatenating them together.
  """
  assert os.path.isdir(dir_path)
  datasets = []
  n_loaded, n_skipped = 0, 0
  tick = time.time()
  for file_name in os.listdir(dir_path):
    try:
      dataset = LocalEnvironmentDataSet(
        os.path.join(dir_path, file_name), cache_dir=cache_dir)
      datasets.append(dataset)
      n_loaded += 1
    except ValueError:
      logger.warning(f"Could not load {file_name}")
      n_skipped += 1
  tock = time.time()
  output = ConcatDataset(datasets)
  logger.info(f"Loaded {n_loaded} files, skipped {n_skipped} files " \
              f"in {tock - tick:.2f} seconds; {len(output)} residues collected.")
  return output


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