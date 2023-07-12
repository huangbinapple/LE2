import argparse
import os
import json
import time
import torch
from torch.utils.data import DataLoader
from le2.common import log
from le2.model.modules import ResidueTypePredictor
from le2.data.dataset import LocalEnvironmentDataSet, collate_fn, \
  construct_dataset_from_dir

logger = log.logger


def evaluate(model, dataloader, **kwargs):
  """Evaluate a PyTorch model on a given dataset.

  Args:
    - model (torch.nn.Module): The PyTorch model to evaluate.
    - dataloader (torch.utils.data.DataLoader): The PyTorch dataloader to use
      for evaluation.
    - kwargs: Additional keyword arguments to pass to the model.
      See le2/model/modules.ResidueTypePredictor.forward for more details.

  Returns:
    Tuple[float, float]: A tuple containing the average evaluation loss and accuracy.
  """
  # Set the model to evaluation mode.
  model.eval()
  logger.info('Start evaluation ...')
  tick = time.time()
  print_timer = time.time()

  # Disable gradient computation and tracking during evaluation.
  with torch.no_grad():
    output = {}
    validate_nsample = 0
    is_first_batch = True
    for batch in dataloader:
      result = model(batch, **kwargs)
      # Get a arbitrary value from the result dict.
      value = next(iter(result.values()))
      validate_nsample += len(value)

      if is_first_batch:
        for key in result:
          output[key] = []
        is_first_batch = False
        
      for key in result:
        output[key].append(result[key])
        
      if time.time() - print_timer > 10:
        logger.info(f'{validate_nsample} samples processed')
        print_timer = time.time()
        
    for key in output:
      output[key] = torch.cat(output[key], dim=0)
      
    logger.info(f'Evaluate on {validate_nsample} samples '
                        f'in {time.time() - tick:.2f} seconds.')
    return output

def main(args):
  # Create model.
  input_dim = 46 if args.senpai else 45
  model = ResidueTypePredictor(
    input_dim, args.d_model, args.nhead, args.nlayer, args.device)
  # Load model parameters.
  state_dict = torch.load(args.model_path, map_location=args.device)['state_dict']
  model.load_state_dict(state_dict)
  # Create dataloader.
  logger.info(f'Loading data from {args.evaluate_path} ...')
  if os.path.isfile(args.evaluate_path):
    dataset = LocalEnvironmentDataSet(args.evaluate_path, radius=12)
  else:
    dataset = construct_dataset_from_dir(args.evaluate_path, cache=args.cache, radius=12)
  dl = DataLoader(dataset, batch_size=4096 * 2, collate_fn=collate_fn,
                  num_workers=args.nworker)
  # Evaluate model.
  output = evaluate(model, dl, output_iscorrect=True, senpai=args.senpai)
  print(output['iscorrect'].sum().item() / len(output['iscorrect']))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  ## Dataset parameters
  parser.add_argument('-E', '--evaluate_path', type=str,
                      help='File or directory to evaluate')
  parser.add_argument('-c', '--cache', action='store_true',
                      help='Create/load cache file for faster loading')
  parser.add_argument('--no-cache', action='store_false', dest='cache',
                      help='Do not create/load cache file for faster loading')
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
  n_work_default = max(8, os.cpu_count())
  parser.add_argument('-N', '--nworker', type=int, default=n_work_default,
                      help='Number of workers, default: max(8, cpu_count)')
  parser.add_argument('-l', '--log_level', type=str, default='INFO')
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
  
  # Check evaluate_path is specified.
  if args.evaluate_path is None:
    raise ValueError('Please specify the path to training and evaluation data.')
  if args.model_path is None:
    raise ValueError('Please specify the path to the model to evaluate.')
  args.evaluate_path = os.path.realpath(args.evaluate_path)
  if args.config:
    args.config = os.path.realpath(args.config)
  
  # Set gloabl log level=
  logger.setLevel(args.log_level)
  # Set stream handler
  log.add_stream_handler(logger, args.log_level)

  main(args)