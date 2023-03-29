from torch.utils.data import Dataset
from le2.common.protein import Protein


class LocalEnvironmentDataSet(Dataset):
  def __init__(self, file_path):
    self.file_path = file_path
    with open(file_path) as f:
      self.protien = Protein(f.read())
    
  def __len__(self):
    return len(self.protien)
  
  def __getitem__(self, index):
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
      - protein (le2.common.protein.Protein): the protein object
    """
    features = {}
    neighbor_indicies = self.protien.get_neighbor_indicies(index)
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
    
    meta = dict(file_path=self.file_path, protein=self.protien)
    return {'feature': features, 'label': label, 'meta': meta}
      