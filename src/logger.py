import logging

def get_logger() -> logging.Logger:
    """
    Returns a logger:
        logging.logger
    """
    logger = logging.getLogger('dataflow')
    logger.setLevel(logging.INFO)
    return logger