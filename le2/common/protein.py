"""Protein data type."""
import logging
import io
import Bio  # type: ignore
from Bio.PDB import PDBParser, FastMMCIFParser  # type: ignore
import torch
from le2.common import log
from le2.common import utils, r3

logger = log.logger


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
  main_chain_atom_coords = []
  residue_names = []
  residue_indices = []
  chain_ids = []

  # Loop through each residue in the structure
  for residue in model.get_residues():
    if residue.id[0] != ' ':  # Ignore hetero residues
      continue
    if residue.id[2] != ' ':  # Encountered an insertion code
      raise ValueError('Encountered an insertion code {} in chain {}'.format(
        residue.id, residue.get_parent().id))
    
    residue_atom_coords = []

    # Get the coordinates of the CA, C, and N atoms for the residue
    for atom_name in ['N', 'CA', 'C']:
      try:
        atom = residue[atom_name]
        coords = atom.get_coord().tolist()
      except KeyError:
        coords = [float('nan')] * 3
      residue_atom_coords.append(coords)

    main_chain_atom_coords.append(residue_atom_coords)
    residue_names.append(residue.get_resname())
    residue_indices.append(residue.get_id()[1])
    chain_ids.append(residue.get_parent().id)

  main_chain_atom_coords = torch.tensor(main_chain_atom_coords)

  # Create the output dictionary
  output = {
      'main_chain_atoms': main_chain_atom_coords,  # shape: (L, 3, 3)
      'residue_names': residue_names,  # elements: str
      'residue_indices': residue_indices,  # elements: int
      'chain_ids': chain_ids,  # elements: str
      'multiple_models': multiple_models,  # bool
  }

  return output


class Protein:
  """Protein structure representation."""
  
  big_distance = 15.0  # Angstroms

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
        nonzero().squeeze(-1)
    self.atom_coords =\
      self._atoms_info['main_chain_atoms'][self._valid_residues_index]
      # shape: (L, 3, 3)
    self.residue_frames = r3.vec2transform(self.atom_coords)  # shape: (L, 4, 4)
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
    # Compress mutual_ca_distances to save disk space.
    state['mutual_ca_distances'] = \
      utils.convert_to_sparse_coo(self.mutual_ca_distances, self.big_distance)
    del state['_structure']
    del state['_raw_string']
    del state['_file_type']
    del state['_atoms_info']
    del state['_valid_residues_index']
    return state
    
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
  
  def _calculate_mutual_ca_distances(self) -> torch.Tensor:
    """Calculate the distances between all pairs of CA atoms in the protein."""
    # Calculate the distances between all pairs of CA atoms
    ca_atoms = self.residue_frames[:, :3, 3]  # size: (L, 3)
    logger.debug(f'Calculating the distances between all pairs '\
                 f'({len(ca_atoms)}) of CA atoms...')
    distances = torch.cdist(ca_atoms, ca_atoms)
    return distances  # shape: (L, L), type: torch.tensor
  
  def backbone_to_pdb_string(self):
    lines = []
    for i, atom_coords in enumerate(self.atom_coords):
      for atom_name, coords in zip(['N', 'CA', 'C'], atom_coords):
        atom_index = i * 3 + ['N', 'CA', 'C'].index(atom_name) + 1
        lines.append(
          f'ATOM  {atom_index:>5}  {atom_name:<3} {self.residue_names[i]:>3} '\
          f'{self.residue_chain_ids[i]:>1}{self.residue_indicies[i]:>4}    '\
          f'{coords[0]:>8.3f}{coords[1]:>8.3f}{coords[2]:>8.3f}  1.00  '\
          f'0.00          {atom_name[0]:>2}\n')
    return ''.join(lines)
  
  def distance_matrix_to_dense(self) -> None:
    """Convert a sparse CA matrix to a dense one.
    This save training time, but takes up more memory.
    """
    if self.mutual_ca_distances.layout == torch.sparse_coo:
      self.mutual_ca_distances = self.mutual_ca_distances.to_dense()
  
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
    ALMOST_ZERO = 0.1  # cdistance < ALMOST_ZERO is considered as 0
    assert cutoff < self.big_distance, 'The cutoff distance is too big.'
    ca_distances = self.mutual_ca_distances[residue_index]
    if ca_distances.layout == torch.sparse_coo:
      ca_distances = ca_distances.to_dense()
    # Get values's index whose value ins between 0 and cutoff
    index = (ca_distances > ALMOST_ZERO) & (ca_distances < cutoff)
    return index.nonzero().squeeze(-1)


if __name__ == '__main__':
  import doctest
  doctest.testmod()
