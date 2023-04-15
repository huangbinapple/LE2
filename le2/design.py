import argparse
import os
import json
import time
import random
import torch
from torch.utils.data import DataLoader, Subset, SequentialSampler
from le2.common import log
from le2.model.modules import ResidueTypePredictor
from le2.data.dataset import LocalEnvironmentDataSet, collate_fn, \
  construct_dataset_from_dir
from le2.common import residue_constants as rc
from le2.common.utils import FixedOrderSampler

logger = log.logger


class SequenceDesigner():
  
  def __init__(self, model, senpai=False):
    self.radius = 12
    self.model = model
    self.model.eval()
    self.dataset = None
    self.is_correct = None
    self.loss = None
    self.predicted_rtype = None
    self.senpai = senpai
    self._subset_sampler = FixedOrderSampler([])
    
  def _initialize_seq(self):
    """Randomly initialize the sequence."""
    # Set each position to a random residue type.
    for i in range(len(self.seq)):
      self.seq[i] = rc.resnames[random.randint(0, 19)]
    logger.debug('Randomly initialized sequence ...')
    logger.debug(f'current sequence:\t {self.long_to_short_seq(self.seq)}')
    inital_output = self._predict()
    self.is_correct = inital_output['iscorrect']
    self.loss = inital_output['loss']
    self.predicted_rtype = inital_output['predicted_rtype']
    
  def _iter(self, index):
    """Change residues at index to their predicted rtype."""
    if self.dataset is None:
      raise ValueError('No dataset loaded')
    logger.debug("Sequence index: \t " + ''.join(str(i) * 10 for i in range(10)))
    logger.debug("Sequence index: \t " + (''.join(str(i) for i in range(10))) * 10)
    logger.debug(f'current sequence:\t {self.long_to_short_seq(self.seq)}')
    logger.debug(f'predicted sequence:\t '
                 f"{''.join(rc.restypes[i] for i in self.predicted_rtype)}")
    logger.debug(f'index to change:\t {index}')
    # Change residues at index to predicted type and collect affected neighbors.
    for i, rtype_index in zip(index, self.predicted_rtype[index]):
      original_residue = self.seq[i]
      self.seq[i] = rc.resnames[rtype_index]
      logger.debug(
        f"{i}: {rc.restype_3to1[original_residue]}({original_residue}) -> "
        f"{rc.restypes[rtype_index]}({rc.resnames[rtype_index]})")
    logger.debug(f"sequence updated: \t {self.long_to_short_seq(self.seq)}")
    neighbor_index = self.is_neighborhood[index].any(0).nonzero().squeeze(-1)
    self.is_correct[index] = True
    # Update neighbors' states.
    update_output = self._predict(neighbor_index)
    self.is_correct[neighbor_index] = update_output['iscorrect']
    self.loss[neighbor_index] = update_output['loss']
    self.predicted_rtype[neighbor_index] = update_output['predicted_rtype']
    logger.debug(f"Updated {len(neighbor_index)} residues")
    logger.debug(f"loss: {self.get_loss()}; accuracy: {self.get_accuracy()}")
    logger.debug(f"is_correct: \t\t "
                 f"{''.join(map(str, self.is_correct.type(torch.int).tolist()))}")
    logger.debug('\n')
    
  def _predict(self, index=None):
    # indicator = torch.zeros(len(self.dataset), dtype=torch.int)  # For log.
    if self.dataset is None:
      raise ValueError('No dataset loaded')
    if index is None:
      index = range(len(self.dataset))
    # indicator[index] = 1
    # logger.debug(f"Update {len(dataset)} residues: \t "
    #              f"{''.join(map(str, indicator.tolist()))}")
    self._subset_sampler.indices = index
    batch = next(iter(self.dl))
    with torch.no_grad():
      output = self.model(batch, output_iscorrect=True,
                          output_loss=True, output_predicted_rtype=True,
                          senpai=self.senpai)
    return output
  
  def get_loss(self):
    return self.loss.mean().item()
  
  def get_accuracy(self):
    return self.is_correct.float().mean().item()
  
  @staticmethod
  def long_to_short_seq(long_seq):
    """Transform 3 letter sequence to 1 letter sequence."""
    return ''.join(rc.restype_3to1[residue] for residue in long_seq)
    
  def load_file(self, file_path):
    self.protein_name = os.path.basename(file_path).split('.')[0]
    self.dataset = LocalEnvironmentDataSet(file_path)
    self.dl = DataLoader(self.dataset, batch_size=len(self.dataset),
                         sampler=self._subset_sampler, collate_fn=collate_fn)
    self.protein = self.dataset.protein
    self.seq = self.protein.residue_names
    self.original_seq = self.seq.copy()
    ca_distances = self.protein.mutual_ca_distances.to_dense()
    self.is_neighborhood = (ca_distances > 0) & (ca_distances < self.radius)
    
  def design(self, output_path, seed=42):
    """Design the sequence."""
    # Set seed.
    ticker = time.time()
    logger.info(f'Start designing {self.protein_name} ...')
    random.seed(seed)
    self._initialize_seq()
    
    niter = 0
    stage_1_length = int(len(self.seq) / 1)
    stage_2_length = int(len(self.seq) / 5)
    schedule = stage_1_length * [5] + stage_2_length * [1]
    for niter, num_change in enumerate(schedule):
      logger.debug(f"iter: {niter + 1}")
      incorrect_index = (~self.is_correct).nonzero().squeeze(-1).tolist()
      if len(incorrect_index) == 0:
        break
      index_to_change = random.sample(
        incorrect_index, min(len(incorrect_index), num_change))
      self._iter(index_to_change)
    runtime = time.time() - ticker
    logger.info(f"Designed a sequence of length {len(self.seq)} "
                f"in {runtime:.2f} seconds")
    identity = sum((self.seq[i] == self.original_seq[i]
                    for i in range(len(self.seq)))) / len(self.seq)
    output = {'sequence': self.long_to_short_seq(self.seq),
              'loss': self.get_loss(),
              'accuracy': self.get_accuracy(),
              'identity': identity,
              'niter': niter}
    
    # Write design output to fasta file.
    with open(output_path, 'w') as f:
      file_name = os.path.basename(output_path)
      protein_name = os.path.splitext(file_name)[0]
      with open(output_path, 'w') as f:
        f.write(f">{protein_name}, accuracy: {output['accuracy']:.4f}, "
                f"loss: {output['loss']:.4f}, identity: {output['identity']:.4f}, "
                f"run_time: {runtime:.4f}, niter: {output['niter']}, "
                f"seed: {seed}\n")
        f.write(f"{output['sequence']}\n")
    return output


def main(args):
  # print(vars(args))
  # return
  # Create model.
  input_dim = 46 if args.senpai else 45
  model = ResidueTypePredictor(
    input_dim, args.d_model, args.nhead, args.nlayer, args.device)
  # Load model parameters.
  state_dict = torch.load(args.model_path, map_location=args.device)['state_dict']
  model.load_state_dict(state_dict)
  # Start design sequence.
  designer = SequenceDesigner(model, senpai=args.senpai)
  if os.path.isdir(args.target_path):
    for file_name in os.listdir(args.target_path):
      if file_name.endswith('.pdb') or file_name.endswith('.cif'):
        file_path = os.path.join(args.target_path, file_name)
        output_path = os.path.join(
          args.output_path, os.path.splitext(file_name)[0] + '.fasta')
        designer.load_file(file_path)
        designer.design(output_path, args.seed)
  else:
    designer.load_file(args.target_path)
    designer.design(args.output_path, args.seed)

  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  ## Dataset parameters
  parser.add_argument('-T', '--target_path', type=str,
                      help='File or directory to evaluate')
  parser.add_argument('-O', '--output_path', type=str,
                      help='Output path, default: <target>.fasta in target directory')
  parser.add_argument('-S', '--seed', type=int, default=42,
                      help='Random seed, default: 42')
  ## Model parameters
  parser.add_argument('-D', '--d_model', type=int, default=256,
                      help='Model dimension, default: 256')
  parser.add_argument('-H', '--nhead', type=int, default=16,
                      help='Number of heads, default: 16')
  parser.add_argument('-L', '--nlayer', type=int, default=3,
                      help='Number of layers, default: 3')
  parser.add_argument('-M', '--model_path', type=str,
                      help='Path to the model to evaluate')
  parser.add_argument('-d', '--device', type=str, default='cpu',
                      help='Device, default: cpu')
  parser.add_argument('--senpai', action='store_true', default=False,
                      help='Use senpai model, default: False')
  parser.add_argument('--no-senpai', dest='senpai', action='store_false',
                      help='Use normal model')
  ## Other parameters
  parser.add_argument('-l', '--log_level', type=str, default='INFO')
  parser.add_argument('--log_file', type=str)
  parser.add_argument('-C', '--config', type=str,
                      help='Path to JSON config file')
  
  # Parse arguments for the first time.
  args = parser.parse_args()
  # Load config file for overwriting arguments.
  if args.config:
    with open(args.config, 'r') as f:
        config = json.load(f)
        config = {k: v for k, v in config.items() if k in args}
    parser.set_defaults(**config)
  # Parse arguments again for overwriting config file.
  args = parser.parse_args()
  
  # Check target_path is specified.
  if args.target_path is None:
    raise ValueError('Please specify the path to training and evaluation data.')
  if args.model_path is None:
    raise ValueError('Please specify the path to the model to evaluate.')
    
  # Set output path default.
  if args.output_path is None:
    if os.path.isdir(args.target_path):
      args.output_path = args.target_path
    else:
      args.output_path = os.path.splitext(args.target_path)[0] + '.fasta'
      
  # Make output directory.
  if os.path.isdir(args.target_path):
    os.makedirs(args.output_path, exist_ok=True)
    
  # Get real path.
  args.target_path = os.path.realpath(args.target_path)
  if args.config:
    args.config = os.path.realpath(args.config)
  
  # Set gloabl log level=
  logger.setLevel(args.log_level)
  # Set stream handler
  log.add_stream_handler(logger, args.log_level)
  # Set file handler
  if args.log_file:
    log.add_file_handler(logger, args.log_file, args.log_level)

  main(args)