"""Protein data type."""
import logging
import io
import Bio  # type: ignore
from Bio.PDB import PDBParser, FastMMCIFParser  # type: ignore
import torch
from le2 import config


logger = logging.getLogger(config.LOG_NAME)


def extract_main_chain_atoms(structure: Bio.PDB.Structure.Structure) -> dict:
  """
  Extracts the main chain atoms (N, CA, C) for each residue in a Bio.PDB structure object and
  returns a dictionary containing the main chain atoms as a torch.tensor array of size (L, 3, 3),
  where L is the number of residues in the structure. The second dimension represents the three
  main chain atoms (N, CA, C) and the third dimension represents their (x, y, z) coordinates.
  Also returns three lists of length L, storing the residue name, residue index, and chain ID,
  respectively.

  If any of the main chain atoms are missing for a residue, their coordinates will be set to NaN.

  Args:
  - structure (Bio.PDB.Structure): a Bio.PDB structure object containing a protein structure

  Returns:
  - output (dict): a dictionary containing the following fields:
    - main_chain_atoms (torch.tensor): a torch.tensor array of size (L, 3, 3) containing the
        coordinates of the main chain atoms (N, CA, C) for each residue in the structure
    - residue_names (list): a list of length L containing the residue names for each residue in
        the structure
    - residue_indices (list): a list of length L containing the residue indices for each residue in
        the structure
    - chain_ids (list): a list of length L containing the chain IDs for each residue in the structure
    - multiple_models (bool): a boolean flag indicating whether the structure contains multiple models (True) or not (False)
  """
  # Check if the structure contains multiple models
  num_models = len(structure)
  multiple_models = num_models > 1
  
  # Select the first model if there are multiple models
  if multiple_models:
    model = structure[0]
  else:
    model = structure

  # Initialize the lists for the output fields
  main_chain_atoms = []
  residue_names = []
  residue_indices = []
  chain_ids = []

  # Loop through each residue in the structure
  for residue in model.get_residues():
    if residue.id[0] != ' ':  # Ignore hetero residues
      continue
    if residue.id[2] != ' ':  # Encountered an insertion code
      raise ValueError('Encountered an insertion code: {} in chain {}'.format(
        residue.id, residue.get_parent().id))
    
    residue_atoms = []

    # Get the coordinates of the CA, C, and N atoms for the residue
    for atom_name in ['N', 'CA', 'C']:
      try:
        atom = residue[atom_name]
        coords = torch.tensor(atom.get_coord())
      except KeyError:
        coords = torch.tensor([float('nan')] * 3)
      residue_atoms.append(coords)

    main_chain_atoms.append(residue_atoms)
    residue_names.append(residue.get_resname())
    residue_indices.append(residue.get_id()[1])
    chain_ids.append(residue.get_parent().id)

  main_chain_atoms = torch.stack(
    [torch.stack(residue_atoms) for residue_atoms in main_chain_atoms])

  # Create the output dictionary
  output = {
      'main_chain_atoms': main_chain_atoms,  # shape: (L, 3, 3)
      'residue_names': residue_names,  # elements: str
      'residue_indices': residue_indices,  # elements: int
      'chain_ids': chain_ids,  # elements: str
      'multiple_models': multiple_models,  # bool
  }

  return output


class Protein:
  """Protein structure representation."""

  def __init__(self, raw_string: str, file_type: str ='pdb'):
    """Initialize a Protein object.

    Args:
    - raw_string (str): a string containing the path to the input file
    - file_type (str): the type of the input file, either 'pdb' or 'cif'
    """
    self._raw_string = raw_string
    self._file_type = file_type
    self._structure = self._load_structure()
    self._atoms_info = extract_main_chain_atoms(self._structure)
    self._valid_residues_index =\
      (~self._atoms_info['main_chain_atoms'].sum(dim=(1, 2)).isnan()).\
        nonzero().squeeze()
    self.atom_coords =\
      self._atoms_info['main_chain_atoms'][self._valid_residues_index]
      # shape: (L, 3, 3)
    self.residue_names =\
      [self._atoms_info['residue_names'][i] for i in self._valid_residues_index]
    self.residue_indicies =\
      [self._atoms_info['residue_indices'][i] for i in self._valid_residues_index]
    self.residue_chain_ids =\
      [self._atoms_info['chain_ids'][i] for i in self._valid_residues_index]
      
    # Calculate the distances between all pairs of CA atoms in 3D space
    self.mutual_ca_distances = self._calculate_mutual_ca_distances()
    # Shape: (L, L)
    
  def __getstate__(self):
    state = self.__dict__.copy()
    del state['_structure']
    del state['_raw_string']
    del state['_file_type']
    del state['_atoms_info']
    del state['_valid_residues_index']
    return state
    
  def __setstate__(self, state):
    self.__dict__.update(state)
    
  def __len__(self) -> int:
    """Return the number of residues in the protein."""
    return len(self.atom_coords)
    
  def _load_structure(self) -> Bio.PDB.Structure.Structure:
    """Load the protein structure from the input file."""
    if self._file_type == 'pdb':
      parser = PDBParser(QUIET=True)
    elif self._file_type == 'cif':
      parser = FastMMCIFParser(QUIET=True)
    else:
      raise ValueError('Invalid file type: {}'.format(self._file_type))
    handle = io.StringIO(self._raw_string)
    return parser.get_structure('protein', handle)
  
  def _calculate_mutual_ca_distances(self) -> torch.tensor:
    """Calculate the distances between all pairs of CA atoms in the protein."""
    # Calculate the distances between all pairs of CA atoms
    ca_atoms = self.atom_coords[:, 1, :]  # size: (L, 3)
    logger.debug(f'Calculating the distances between all pairs '\
                 f'({len(ca_atoms)}) of CA atoms...')
    distances = torch.cdist(ca_atoms, ca_atoms)
    return distances  # shape: (L, L)
  
  def get_neighbor_indicies(
      self, residue_index: int, cutoff: float =12.0) -> torch.tensor:
    """Get the neighbors of a residue.

    Args:
    - residue_index (int): the index of the residue
    - cutoff (float): the cutoff distance for defining a neighbor

    Returns:
    - neighbors (torch.tensor): a troch tensor of type int containing the
    indices of the neighboring redidues
    """
    # Add cutoff to the diagonal to avoid self-neighborship
    ca_distances = self.mutual_ca_distances +\
      torch.eye(self.mutual_ca_distances.shape[0]) * cutoff
    return (ca_distances[residue_index] < cutoff).nonzero().squeeze()


if __name__ == '__main__':
  import doctest
  doctest.testmod()
