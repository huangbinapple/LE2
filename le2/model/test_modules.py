import unittest
from torch.utils.data import DataLoader
from le2.model.modules import ResidueTypePredictor
from le2.data.dataset import LocalEnvironmentDataSet, collate_fn

class TestResidueTypePredictor(unittest.TestCase):
  
  def setUp(self):
    dataset = LocalEnvironmentDataSet('example/0089.pdb')
    self.sample = next(iter(
      DataLoader(dataset, batch_size=2, collate_fn=collate_fn)))

  def test_forward_pass(self):
    # Create a ResidueTypePredictor instance.
    model = ResidueTypePredictor(d_input=45, d_model=256, n_heads=4, n_layers=3,
                                 device='cpu')

    # Run the forward pass.
    # print(self.sample)
    output = model(self.sample)
    # print(output)

    # Check that the output has the correct shape.
    self.assertEqual(output['logits'].shape, (2, 21))

if __name__ == '__main__':
    unittest.main()