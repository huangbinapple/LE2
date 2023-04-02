import logging


LOG_FILE_PATH = 'log/log.txt'
LOG_LEVEL = logging.DEBUG


# Configure logging
logging.basicConfig(filename=LOG_FILE_PATH, level=LOG_LEVEL,
                    format='%(asctime)s %(levelname)s %(module)s: %(message)s')