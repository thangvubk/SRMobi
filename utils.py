import logging


def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = logging.getLogger('SRMobi')
    if logger.hasHandlers():
        return logger
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)
    return logger
