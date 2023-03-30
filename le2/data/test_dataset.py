import unittest
from torch.utils.data import DataLoader
from le2.common.protein import Protein
from le2.data.dataset import LocalEnvironmentDataSet, collate_fn


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
      
class TestCollateFn(unittest.TestCase):
  def setUp(self):
    file_path = 'example/0089.pdb'
    self.dataset = LocalEnvironmentDataSet(file_path)
    
  def test_collate_fn(self):
    
    # Initialize a DataLoader with batch size 2 and the collate_fn
    dataloader = DataLoader(self.dataset, batch_size=2, collate_fn=collate_fn)
    
    # Iterate through the DataLoader and check the output
    batch = next(iter(dataloader))
    # print(batch)
    self.assertIsInstance(batch, dict)
    self.assertIn('feature', batch)
    self.assertIn('label', batch)
    self.assertIn('meta', batch)
    self.assertIsInstance(batch['feature'], dict)
    self.assertIsInstance(batch['label'], dict)
    self.assertIsInstance(batch['meta'], dict)
    self.assertEqual(len(batch['feature']['neighbor_names']), 2)
    self.assertEqual(len(batch['label']['target_name']), 2)
    self.assertEqual(len(batch['meta']['file_path']), 2)