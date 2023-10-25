import logging
import argparse
import time
import os
import torch
import pickle
from torch.utils.data import Dataset, ConcatDataset
from le2.common.protein import Protein
from le2.common import residue_constants as rc
from le2.common import log, r3
from torch.nn.utils.rnn import pad_sequence

logger = log.logger


class LocalEnvironmentDataSet(Dataset):
  """Local environment dataset."""
  def __init__(self,
               file_path: str,  # Path to the PDB/CIF file
               radius: float =12.0,  # Radius of the local environment
               cache: bool=False,
               noise_level: float=0.0):  # Path to the cache directory
    logger.debug(f"Loading data from {file_path}..., radius={radius}")
    self.radius = radius
    self.file_path = file_path
    self.noise_level = noise_level
    file_type = file_path.split('.')[-1]
    # cache_dir is the directory of the file path.
    cache_dir = os.path.dirname(file_path)
    cache_file = os.path.join(cache_dir, os.path.basename(file_path) + '.pkl')
    # If cache file exists, load from it.
    if cache and os.path.exists(cache_file):
      logger.debug(f"Loading cached data from {cache_file}")
      with open(cache_file, 'rb') as f:
        self.protein = pickle.load(f)
    else:
      # Otherwise, load from the file.
      with open(file_path) as f:
        self.protein = Protein(f.read(), file_type)
      if cache:  # cache option is enabled
        # Save the data to cache file.
        logger.debug(f"Saving data to {cache_file}")
        with open(cache_file, 'wb') as f:
          pickle.dump(self.protein, f)
    # Check if protein has property `residue_frames`.
    if not hasattr(self.protein, 'residue_frames') and noise_level == 0:
      self.protein.residue_frames = r3.vec2transform(self.protein.atom_coords)
    logger.debug(f"Loaded {len(self.protein)} residues from {file_path}")
    
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
      - neighbor_residue_frames (tensor): a tensor of residue frames
      - target_index (int): the index of the target residue
      - target_chain_id (str): the chain id of the target residue
      - target_residue_frames (tensor): a tensor of residue frames
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
    features['neighbor_residue_frames'] =\
      self.protein.residue_frames[neighbor_indicies]
    features['target_index'] = self.protein.residue_indicies[index]
    features['target_chain_id'] = self.protein.residue_chain_ids[index]
    
    if self.noise_level == 0:
      features['target_residue_frames'] = self.protein.residue_frames[index]
      features['neighbor_residue_frames'] =\
        self.protein.residue_frames[neighbor_indicies]
    else:
      residue_frames_noise = r3.vec2transform(
        self.protein.atom_coords +\
          self.noise_level * torch.randn_like(self.protein.atom_coords))
      features['target_residue_frames'] = residue_frames_noise[index]
      features['neighbor_residue_frames'] =\
        residue_frames_noise[neighbor_indicies]
    
    label = dict(target_name=self.protein.residue_names[index])
    
    meta = dict(file_path=self.file_path)
    return {'feature': features, 'label': label, 'meta': meta}
  
  
def construct_dataset_from_dir(
  dir_path: str,
  radius: float =12.0,
  cache: bool=False,
  fast_mode: bool=True,
  noise_level: float=0.0) -> ConcatDataset:
  """
  Construct a ConcatDataset from a directory using all cif and pdb
  files in that dir, by constructing a LocalEnvironmentDataSet for each
  of those files, and then Concatenating them together.
  """
  assert os.path.isdir(dir_path)
  datasets = []
  n_loaded, n_skipped = 0, 0
  tick = time.time()
  for file_name in filter(
      lambda x: x.endswith('.cif') or x.endswith('.pdb'),
      os.listdir(dir_path)):
    try:
      dataset = LocalEnvironmentDataSet(
        os.path.join(dir_path, file_name), radius=radius, cache=cache,
        noise_level=noise_level)
      if fast_mode:
        dataset.protein.distance_matrix_to_dense()
      datasets.append(dataset)
      n_loaded += 1
    except ValueError as e:
      logger.warning(f"Could not load {file_name}: {e}")
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
      - neighbor_residue_frames (torch.tensor):
        shape: (batch_size, max_len, 4, 4)
      - target_index (torch.tensor): shape: (batch_size)
      - target_chain_id (torch.tensor): shape: (batch_size)
      - target_residue_frames (torch.tensor): shape: (batch_size, 4, 4)
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
  
  output['features']['neighbor_residue_frames'] = pad_sequence(
    [sample['feature']['neighbor_residue_frames']
     for sample in batch], batch_first=True)
  
  output['features']['target_index'] = torch.tensor(
    [sample['feature']['target_index'] for sample in batch])
  
  output ['features']['target_chain_id'] = torch.tensor(
    list(map(hash, [sample['feature']['target_chain_id'] for sample in batch])))

  output['features']['target_residue_frames'] = torch.stack(
    [sample['feature']['target_residue_frames'] for sample in batch])
  
  output['labels']['target_name'] = torch.tensor(
    [rc.get_resname_index(sample['label']['target_name']) for sample in batch])
    
  output['meta']['file_path'] = [sample['meta']['file_path'] for sample in batch]
  
  return output


if __name__ == '__main__':
  # Create cache for a directory
  parser = argparse.ArgumentParser(
    description='Create cache for a directory')
  parser.add_argument('protein_dir', type=str, help='Path to directory')
  parser.add_argument('cache_dir', type=str, help='Path to cache directory')
  parser.add_argument('--radius', type=float, default=12.0,
                      help='Radius of local environment')
  parser.add_argument('--log', type=str)
  parser.add_argument('--log_level', type=str, default='INFO',)
  args = parser.parse_args()
  
  logger.setLevel(args.log_level)
  # Set up logging
  if args.log is None:
    logger.addHandler(logging.NullHandler())
  else:
    log.add_file_handler(logger, args.log, args.log_level)
  
  construct_dataset_from_dir(
    args.protein_dir, radius=args.radius, cache_dir=args.cache_dir)
  