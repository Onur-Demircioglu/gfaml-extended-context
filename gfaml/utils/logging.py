# gfaml/utils/logging.py
import logging

def get_logger(name: str) -> logging.Logger:
    """
    Araştırma deposu için standart konsol kaydedici (logger) döner.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
