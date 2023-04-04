import unittest
import torch
from le2.data.dataset import LocalEnvironmentDataSet, collate_fn
from torch.utils.data import DataLoader
from le2.model.feature import make_feature


# Test make_feature
class TestMakeFeature(unittest.TestCase):
  
  def setUp(self):
    self.dataset = LocalEnvironmentDataSet('example/0089.pdb')
    self.sample = next(iter(DataLoader(
      self.dataset, batch_size=2, collate_fn=collate_fn)))
    
  # Test the output shape of make_feature
  def test_shape(self):
    feature = make_feature(self.sample['features'])
    max_length = self.sample['features']['neighbor_names'].shape[1]
    self.assertEqual(feature.shape, (2, max_length, 45))
    self.assertIsInstance(feature.dtype, torch.float32)
