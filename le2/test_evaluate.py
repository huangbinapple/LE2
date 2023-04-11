import unittest

from le2.evaluate import evaluate
from le2.data.dataset import LocalEnvironmentDataSet, collate_fn
from le2.model.modules import ResidueTypePredictor

from torch.utils.data import DataLoader


class EvaluateTest(unittest.TestCase):
  
  def setUp(self):
    test_file = 'example/0089.pdb'
    self.dataset = LocalEnvironmentDataSet(test_file)
    self.dl = DataLoader(self.dataset, batch_size=4, collate_fn=collate_fn)
    self.model = ResidueTypePredictor(45)

  def test_evaluate(self):
    expected_length = len(self.dataset)
    output = evaluate(
      self.model, self.dl, output_loss=True, output_iscorrect=True)
    self.assertEqual(output['loss'].shape, (expected_length,))
    self.assertEqual(output['iscorrect'].shape, (expected_length,))
    
