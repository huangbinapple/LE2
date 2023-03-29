import torch
import unittest
from Bio.PDB import PDBParser, FastMMCIFParser
import numpy as np
from le2.common.protein import extract_main_chain_atoms, Protein


class TestExtractMainChainAtoms(unittest.TestCase):

  def setUp(self):
    # Load the example PDB file
    self.structure_from_pdb = PDBParser(QUIET=True).get_structure(
      'example', 'example/0089.pdb')
    self.structure_from_cif = FastMMCIFParser(QUIET=True).get_structure(
      'example', 'example/1K12.cif')
    self.structure_from_cif_multi_model = FastMMCIFParser(QUIET=True).get_structure(
      'example', 'example/1E17.cif')
    self.structure_from_cif_insertion_code = FastMMCIFParser(QUIET=True).get_structure(
      'example', 'example/1KB5.cif')

  def test_pdb_as_input(self):
    # Extract output from output dict of extract_main_chain_atoms
    main_chain_atom_infos = extract_main_chain_atoms(self.structure_from_pdb)
    main_chain_atoms = main_chain_atom_infos['main_chain_atoms']
    residue_names = main_chain_atom_infos['residue_names']
    residue_indices = main_chain_atom_infos['residue_indices']
    chain_ids = main_chain_atom_infos['chain_ids']
    is_multiple_models = main_chain_atom_infos['multiple_models']

    # Check that the main chain atoms tensor has the correct shape
    self.assertEqual(main_chain_atoms.shape, (100, 3, 3))

    # Check that the first residue is Glycine (GLY)
    self.assertEqual(residue_names[0], 'SER')

    # Check that the first residue index is 1
    self.assertEqual(residue_indices[0], 1)

    # Check that the first residue is in chain A
    self.assertEqual(chain_ids[0], 'A')

    # Check that the coordinates of the first residue's CA atom are correct
    self.assertTrue(torch.allclose(main_chain_atoms[0][1],
                                   torch.tensor([2.687, -0.122, 0.339])))

    # Check that the coordinates of the first residue's C atom are correct
    # (missing atom in the PDB file)
    assert torch.isnan(main_chain_atoms[0][2]).all()

    # Check that the coordinates of the first residue's N atom are correct
    self.assertTrue(torch.allclose(main_chain_atoms[0][0],
                                   torch.tensor([1.240, -0.218, 0.229])))
    
    # Check that the structure has only one model
    self.assertTrue(is_multiple_models == False)
    
  
  def test_cif_as_input(self):

    # Extract output from output dict of extract_main_chain_atoms.
    main_chain_atom_infos = extract_main_chain_atoms(self.structure_from_cif)
    main_chain_atoms = main_chain_atom_infos['main_chain_atoms']
    residue_names = main_chain_atom_infos['residue_names']
    residue_indices = main_chain_atom_infos['residue_indices']
    chain_ids = main_chain_atom_infos['chain_ids']
    is_multiple_models = main_chain_atom_infos['multiple_models']

    # Check that the main chain atoms tensor has the correct shape
    self.assertEqual(main_chain_atoms.shape, (158, 3, 3))

    # Check that the first residue is Glycine (GLY)
    self.assertEqual(residue_names[0], 'VAL')

    # Check that the first residue index is 1
    self.assertEqual(residue_indices[0], 1)

    # Check that the first residue is in chain A
    self.assertEqual(chain_ids[0], 'A')

    # Check that the coordinates of the first residue's CA atom are correct
    self.assertTrue(torch.allclose(main_chain_atoms[0][1], torch.tensor(
      [-18.860, 19.552, 33.174])))

    # Check that the coordinates of the first residue's C atom are correct
    # (missing atom in the PDB file)
    assert torch.isnan(main_chain_atoms[0][2]).all()

    # Check that the coordinates of the first residue's N atom are correct
    self.assertTrue(torch.allclose(main_chain_atoms[0][0], torch.tensor(
      [-19.818, 18.723, 32.377])))
    
    # Check that the structure has only one model
    self.assertTrue(is_multiple_models == False)
    
  def test_multi_model_cif(self):
    
    # Extract output from output dict of extract_main_chain_atoms
    main_chain_atom_infos = extract_main_chain_atoms(
      self.structure_from_cif_multi_model)
    main_chain_atoms = main_chain_atom_infos['main_chain_atoms']
    is_multiple_models = main_chain_atom_infos['multiple_models']

    # Check that the main chain atoms tensor has the correct shape
    self.assertEqual(main_chain_atoms.shape, (90, 3, 3))
    
    # Check that the structure has multiple models.
    self.assertTrue(is_multiple_models == True)
    
  def test_cif_with_insertion_code(self):
    with self.assertRaises(ValueError):
      extract_main_chain_atoms(self.structure_from_cif_insertion_code)


class TestProtein(unittest.TestCase):
  
  def setUp(self):
    pdb_file = "example/0089.pdb"
    with open(pdb_file, 'r') as f:
      raw_string = f.read()
    self.protein = Protein(raw_string, file_type="pdb")

  def test_load_structure(self):
    # Check that the structure is loaded correctly
    self.assertIsNotNone(self.protein._structure)

  def test_get_neighbors(self):
    # Check that the neighbors are correctly computed
    test_i = 10
    neighbors_5 = self.protein.get_neighbor_indicies(test_i, cutoff=5)
    self.assertIn(test_i - 1, neighbors_5)
    self.assertIn(test_i + 1, neighbors_5)
    self.assertNotIn(test_i, neighbors_5)
    neighbors_10 = self.protein.get_neighbor_indicies(test_i, cutoff=10)
    # print('number of neighbors 5: ', neighbors_5.size())
    # print('number of neighbors 10: ', neighbors_10.size())
    self.assertTrue(neighbors_5.size() < neighbors_10.size())

  def test_ca_distance(self):
    # Check that the CA distance is correctly computed
    CA0_coords = [2.687, -0.122, 0.339]
    CA1_coords = [4.911, 2.776, 1.347]
    true_distance =\
      np.linalg.norm(np.array(CA0_coords) - np.array(CA1_coords))
    distance = self.protein.mutual_ca_distances[0, 1].item()
    self.assertAlmostEqual(distance, true_distance, places=1)
