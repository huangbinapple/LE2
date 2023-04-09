import logging


# Create a logger.
LOG_NAME = 'le2'
logger = logging.getLogger(LOG_NAME)

formatter = logging.Formatter(
  '%(asctime)s\t%(name)s@%(module)-10s\t%(levelname)-8s%(message)s')

def add_file_handler(logger, log_file, log_level):
  fh = logging.FileHandler(log_file)
  fh.setLevel(log_level)
  fh.setFormatter(formatter)
  logger.addHandler(fh)