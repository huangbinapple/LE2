import unittest
from le2.common.protein import Protein
from le2.data.dataset import LocalEnvironmentDataSet


class TestLocalEnvironmentDataSet(unittest.TestCase):
  def setUp(self):
    self.file_path = 'example/0089.pdb'
    # Create a test protein object
    with open(self.file_path) as f:
      raw_data = f.read()
      self.protein = Protein(raw_data)
    self.dataset = LocalEnvironmentDataSet(self.file_path)
  
  def test_len(self):
    self.assertEqual(len(self.dataset), len(self.protein))
  
  def test_getitem(self):
    # Test for a valid index
    sample = self.dataset[0]
    self.assertIsInstance(sample, dict)
    self.assertIn('feature', sample)
    self.assertIn('label', sample)
    self.assertIn('meta', sample)
    self.assertIsInstance(sample['feature'], dict)
    self.assertIsInstance(sample['label'], dict)
    self.assertIsInstance(sample['meta'], dict)
    self.assertIn('neighbor_names', sample['feature'])
    self.assertIn('neighbor_indicies', sample['feature'])
    self.assertIn('neighbor_chain_ids', sample['feature'])
    self.assertIn('neighbor_atom_coordinates', sample['feature'])
    self.assertIn('target_index', sample['feature'])
    self.assertIn('target_chain_id', sample['feature'])
    self.assertIn('target_atom_coordinates', sample['feature'])
    self.assertIn('target_name', sample['label'])
    self.assertIn('file_path', sample['meta'])
    
    # print(sample)
    
    # Test for an invalid index
    with self.assertRaises(IndexError):
      sample = self.dataset[len(self.protein) + 1]