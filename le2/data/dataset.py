from torch.utils.data import Dataset
from le2.common.protein import Protein
from le2.common import residue_constants as rc

class LocalEnvironmentDataSet(Dataset):
  def __init__(self, file_path: str, radius: float =12.0):
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
      - meta (dict): 
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
  """
  features = {}
  labels = dict(target_name=[])
  metas = {}
  for sample in batch:
    for key in sample['feature'].keys():
      if key not in features:
        features[key] = []
      if key == 'neighbor_names':
        # Convert residue names to vocab indicies
        features[key].append([rc.get_resname_index(name)
                              for name in sample['feature'][key]])
      elif key == 'target_chain_id':
        # Convert chain id to its hash.
        features[key].append(hash(sample['feature'][key]))
      elif key == 'target_chain_ids':
        features[key].append([hash(chain_id)
                              for chain_id in sample['feature'][key]])
      else:
        features[key].append(sample['feature'][key])
        
    labels['target_name'].append(
      rc.get_resname_index(sample['label']['target_name']))
    
    for key in sample['meta'].keys():
      if key not in metas:
        metas[key] = []
      metas[key].append(sample['meta'][key])
  
  return {'feature': features, 'label': labels, 'meta': metas}