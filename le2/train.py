import os
import argparse
import time
import json
import logging

import torch
from torch.utils.data import DataLoader

from le2.data.dataset import construct_dataset_from_dir
from le2.data.dataset import collate_fn
from le2.model.modules import ResidueTypePredictor
from le2.common import log
from le2.common import utils

logger = log.logger

  
def save_model(
  model, global_step, val_acc, best_models, nsave=3, save_model_dir='.') -> bool:
  """Save model to disk if it is one of the best three models, and remove 
  the worst model if there are more than three models saved.
  Also, save the latest model to disk.
  Saved file will have the following name format:
    [best|lattest]_{global_step}_{val_acc:.2f}.pth
  Each file is a dictionary with the following keys:
    - state_dict: model state dictionary
    - step: global step
    - val_acc: validation accuracy
    - git_hash: git hash of the current commit
    
  Args:
    - model (nn.Module): model to save
    - global_step (int): global step
    - val_acc (float): validation accuracy
    - best_models (list): list of tuples of (global_step, val_acc)
    - nsave (int): number of best models to save
    - save_model_dir (str): directory to save the model
    
  Returns:
    bool: True if the model is saved as one of the best three models.
  """
  def file_name(step, acc, prefix='best'):
    return f'{prefix}_{step}_{acc * 10000:.0f}.pth'
  
  # Prepare the directory to save the model
  
  to_save = dict(
    state_dict=model.state_dict(), step=global_step, val_acc=val_acc)
  
  # Delete the latest model, the only file .pth starts with 'latest'.
  file_to_delete = \
    [x for x in os.listdir(save_model_dir) if x.startswith('latest')]
  assert len(file_to_delete) <= 1
  if file_to_delete:
    os.remove(os.path.join(save_model_dir, file_to_delete[0]))
  
  # Save the latest model
  latest_model_path = os.path.join(
    save_model_dir, file_name(global_step, val_acc, prefix='latest'))
  torch.save(to_save, latest_model_path)

  saved_as_best = False
  # Save the model as one of the best models if it is one of the best models
  if len(best_models) < nsave or val_acc > best_models[-1][1]:
    # Save the model as one of the best models
    new_best_model_path = os.path.join(
      save_model_dir, file_name(global_step, val_acc))
    torch.save(to_save, new_best_model_path)
    saved_as_best = True
    # Remove the worst model if there are more than `nbest` models saved
    if len(best_models) == nsave:
      worst_model_steps, worst_model_accs = best_models.pop()
      worst_model_path = os.path.join(
        save_model_dir,
        file_name(worst_model_steps, worst_model_accs))
      os.remove(worst_model_path)
    # Update the best_models list
    best_models.append((global_step, val_acc))
    # Sort the best_models list by validation accuracy, in descending order
    best_models.sort(key=lambda x: x[1], reverse=True)
    # Create a 'best.pth' softlink directing to the best model
    best_model_step, best_model_acc = best_models[0]
    best_soft_link_path = os.path.join(save_model_dir, 'best.pth')
    if os.path.exists(best_soft_link_path):
      os.unlink(best_soft_link_path)
    os.symlink(file_name(best_model_step, best_model_acc), best_soft_link_path)   
  return saved_as_best


def main(args):

  # Log training parameters
  logger.info(f'Arguments: {args}')
  # Write training parameters to json file.
  with open(os.path.join(args.work_dir, 'train_config.json'), 'w') as f:
    json.dump(vars(args), f, indent=2)
  
  # Setting basic trainig parameters
  train_dir = args.train_dir
  validate_dir = args.validate_dir
  device = args.device
  nsave = args.nsave
  d_model = args.d_model
  nhead = args.nhead
  nlayer = args.nlayer
  nepoch = args.nepoch
  nworker = args.nworker
  batch_size = args.batch_size
  work_dir = args.work_dir
  model_store_dir = os.path.join(work_dir, 'models')
  
  # Create the directory to save the model
  os.makedirs(model_store_dir, exist_ok=True)

  # Define the period to print out the training loss and accuracy
  print_period = args.print_period
  # Define the period to validate the model
  validate_period = args.validate_period
  
  # Create training and validation datasets and dataloaders
  dataset_train = construct_dataset_from_dir(
    train_dir, cache_dir='/mnt/sddata/huangbin/pdb40/train_le_cache')
  dataset_validate = construct_dataset_from_dir(
    validate_dir, cache_dir='/mnt/sddata/huangbin/pdb40/validate_le')
  dl_train = DataLoader(
    dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
    num_workers=nworker, pin_memory=True)
  dl_validate = DataLoader(
    dataset_validate, batch_size=batch_size, collate_fn=collate_fn,
    num_workers=nworker, pin_memory=True)
  
  # Define the model
  d_feature = 45
  model = ResidueTypePredictor(d_feature, d_model, nhead, nlayer, device)
  # Prepare model for training.
  optimizer = torch.optim.Adam(model.parameters())
  # Log model parameters number
  nparams = utils.count_parameters(model) / 1024 ** 2  # in Million
  logger.info(f'Model parameters number: {nparams:.2f} Million.')
  
  global_nsample = 0
  stop_train = False
  
  # Set timer for logging.
  log_timer, validate_timer, training_start_time = [time.time()] * 3
  
  # Best models on validation accuracy
  best_models = []
  
  # Start training
  for epoch in range(nepoch):
    
    if stop_train:
      break
    
    logger.info(f'Start epoch {epoch + 1}')
    
    # Define variables for logging.
    local_loss, local_ncorrect, local_nsample = 0, 0, 0
    
    for batch in dl_train:

      # Train the model for one step.
      optimizer.zero_grad()
      result = model(batch, output_loss=True, output_ncorrect=True)
      loss_mean = result['loss'].mean()
      loss_mean.backward()
      optimizer.step()
      
      # Update the variables for logging.
      local_loss += result['loss'].sum().item()
      local_ncorrect += result['ncorrect']
      local_nsample += len(result['loss'])
      
      # Update gloabl variables.
      global_nsample += len(result["loss"])
      
      # Print the loss and accuracy every print_period.
      if time.time() - log_timer > print_period:
        log_timer = time.time()
        # Calculate training loss and accuracy.
        training_loss = local_loss / local_nsample
        training_acc = local_ncorrect / local_nsample
        # Log the training loss and accuracy.
        epoch_portion = global_nsample / len(dataset_train)
        logger.info(f'Epoch: {epoch_portion:.4f} / {nepoch}: '
                    f'nsample_trained: {global_nsample} (+{local_nsample}), '
                    f'training loss: {training_loss:.4f}, '
                    f'training accuracy: {training_acc:.4f}')
        # Reset variables for logging.
        local_loss, local_ncorrect, local_nsample = 0, 0, 0     
      
      # Validate the model every validate_period, also if the number of
      # samples trained is larger than the batch size (This happen when
      # resouces are limited).
      if time.time() - validate_timer > validate_period and\
        local_nsample > batch_size:
        validate_timer = time.time()
        
        # Enter model evaluation mode
        model.eval()
        logger.info('Start validation ...')
        
        with torch.no_grad():
          validate_loss, validate_ncorrect, validate_nsample = 0, 0, 0
          for batch in dl_validate:
            result = model(batch, output_loss=True, output_ncorrect=True)
            validate_loss += result['loss'].sum().item()
            validate_ncorrect += result['ncorrect']
            validate_nsample += len(result['loss'])
          # Calculate the average validation loss and accuracy.
          validation_loss = validate_loss / validate_nsample
          validation_acc = validate_ncorrect / validate_nsample
          # log model performance.
          logger.info(f'Validated on {validate_nsample} samples '
                      f'in {time.time() - validate_timer:.2f} seconds.')
          # log the validation loss and accuracy.
          logger.info(f'Epoch: {epoch_portion:.4f} / {nepoch}: '
                      f'nsample_trained: {global_nsample}, '
                      f'validation loss: {validation_loss:.4f}, '
                      f'validation accuracy: {validation_acc:.4f}')
        # Save model.
        is_new_best = save_model(model, global_nsample, validation_acc,
                                 best_models, nsave=nsave,
                                 save_model_dir=model_store_dir)
        # If the model is not saved as one of the best models, and the autostop
        # option is enabled, stop the training.
        if not is_new_best and args.autostop:
          stored_model_accs = [acc for _, acc in best_models]
          logger.info(f'Training Stop, validation accuracy is not improved.'
            f'({validation_acc:.4f} is less than any of {stored_model_accs}.)')
          stop_train = True
          break
        
      # Exit model evaluation mode
      model.train()
    logger.info(f'Training stop, reached the maximum number of epochs.')
      
  logger.info(
    f'Total training time: {time.time() - training_start_time:.2f} s')
      
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  ## Dataset parameters
  parser.add_argument('-T', '--train_dir', type=str,
                      help='Path to training data')
  parser.add_argument('-V', '--validate_dir', type=str,
                      help='Path to validation data')
  ## Model parameters
  parser.add_argument('-D', '--d_model', type=int, default=256,
                      help='Model dimension, default: 256')
  parser.add_argument('-H', '--nhead', type=int, default=16,
                      help='Number of heads, default: 16')
  parser.add_argument('-L', '--nlayer', type=int, default=3,
                      help='Number of layers, default: 3')
  parser.add_argument('-d', '--device', type=str, default='cpu',
                      help='Device, default: cpu')
  ## Training parameters
  parser.add_argument('-A', '--autostop', action='store_true', default=False,
                      help='Stop training if validation accuracy is not improved'
                      '(Worse than 3 stored models), default: False')
  parser.add_argument('-s', '--nsave', type=int, default=5,
                      help='Number of best models to store, default: 5')
  parser.add_argument('-p', '--print_period', type=int, default=10,
                      help='Print period, unit: second, default: 10')
  parser.add_argument('-v', '--validate_period', type=int, default=180,
                      help='Validate period, unit: second, default: 180')
  parser.add_argument('-B', '--batch_size', type=int, default=2048,
                      help='Batch size, default: 2048')
  parser.add_argument('-E', '--nepoch', type=int, default=10,
                      help='Number of epochs, default: 10')
  parser.add_argument('-N', '--nworker', type=int, default=8,
                      help='Number of workers, default: 8')
  ## Other parameters
  parser.add_argument('--config', type=str,
                      help='Path to JSON config file')
  parser.add_argument('--work_dir', type=str, default='new_trainig',
                      help='Path to working directory, default: new_training')
  parser.add_argument('--log_level', type=str, default='INFO')
  
  # Parse arguments for the first time.
  args = parser.parse_args()
  # Load config file for overwriting arguments.
  if args.config:
    with open(args.config, 'r') as f:
        config = json.load(f) 
    parser.set_defaults(**config)
  # Parse arguments again for overwriting config file.
  args = parser.parse_args()
  
  # Turn relative path to absolute path.
  args.train_dir = os.path.realpath(args.train_dir)
  args.validate_dir = os.path.realpath(args.validate_dir)
  args.work_dir = os.path.realpath(args.work_dir)
  if args.config:
    args.config = os.path.realpath(args.config)
  
  os.makedirs(args.work_dir)
  
  # Set gloabl log level
  logger.setLevel(args.log_level)
  # Set stream handler
  log.add_stream_handler(logger, args.log_level)
  # Set file handler
  log_path = os.path.join(args.work_dir, 'train.log')
  log.add_file_handler(logger, log_path, args.log_level)
    
  main(args)
