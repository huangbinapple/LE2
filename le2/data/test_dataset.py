import unittest
import torch
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
    dataset = self.dataset
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    
    lengths = [len(dataset[i]['feature']['neighbor_names']) for i in range(2)]
    # print('lengths:', lengths)
    max_len = max(lengths)
    
    # Check output dictionary keys
    assert set(batch.keys()) == set(['features', 'labels', 'meta', 'mask'])
    
    # Check features dictionary keys
    assert set(batch['features'].keys()) == set([
      'neighbor_names', 'neighbor_indicies', 'neighbor_chain_ids',
      'neighbor_atom_coordinates', 'target_index', 'target_chain_id',
      'target_atom_coordinates'])
    
    # Check mask shape
    print(batch['mask'].shape)
    assert batch['mask'].shape == (2, max_len)
    
    # Check neighbor_names shape and type
    assert batch['features']['neighbor_names'].shape == (2, max_len)
    assert isinstance(batch['features']['neighbor_names'], torch.Tensor)
    
    # Check neighbor_indicies shape and type
    assert batch['features']['neighbor_indicies'].shape == (2, max_len)
    assert isinstance(batch['features']['neighbor_indicies'], torch.Tensor)
    
    # Check neighbor_chain_ids shape and type
    assert batch['features']['neighbor_chain_ids'].shape == (2, max_len)
    assert isinstance(batch['features']['neighbor_chain_ids'], torch.Tensor)
    
    # Check neighbor_atom_coordinates shape and type
    assert batch['features']['neighbor_atom_coordinates'].shape == (
      2, max_len, 3, 3)
    assert isinstance(batch['features']['neighbor_atom_coordinates'], torch.Tensor)
    
    # Check target_index shape and type
    assert batch['features']['target_index'].shape == (2,)
    assert isinstance(batch['features']['target_index'], torch.Tensor)
    
    # Check target_chain_id shape and type
    assert batch['features']['target_chain_id'].shape == (2,)
    assert isinstance(batch['features']['target_chain_id'], torch.Tensor)
    
    # Check target_atom_coordinates shape and type
    assert batch['features']['target_atom_coordinates'].shape == (2, 3, 3)
    assert isinstance(batch['features']['target_atom_coordinates'], torch.Tensor)
    
    # Check labels dictionary keys
    assert set(batch['labels'].keys()) == set(['target_name'])
    
    # Check target_name shape and type
    assert batch['labels']['target_name'].shape == (2,)
    assert isinstance(batch['labels']['target_name'], torch.Tensor)
    
    # Check meta dictionary keys
    assert set(batch['meta'].keys()) == set(['file_path'])
    
    # Check file_path type
    assert isinstance(batch['meta']['file_path'], list)
    assert isinstance(batch['meta']['file_path'][0], str)
    
    print(batch)