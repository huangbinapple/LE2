import logging


LOG_FILE_PATH = 'log/log.txt'
LOG_LEVEL = logging.DEBUG
LOG_NAME = 'le2'

# Create a logger and set its level to DEBUG
logger = logging.getLogger(LOG_NAME)
logger.setLevel(LOG_LEVEL)

# Create a file handler and set its level to DEBUG
fh = logging.FileHandler(LOG_FILE_PATH)
fh.setLevel(LOG_LEVEL)

# Create a formatter and add it to the handler
formatter = logging.Formatter(
  '%(asctime)s %(levelname)s %(module)s: %(message)s')
fh.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(fh)