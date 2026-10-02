import logging
from logging.handlers import RotatingFileHandler
from src.config import LOG_DIR

def get_logger(name:str)->logging.Logger:
 logger=logging.getLogger(name)
 if not logger.handlers:
  logger.setLevel(logging.INFO)
  handler=RotatingFileHandler(LOG_DIR/"application.log",maxBytes=1_000_000,backupCount=3,encoding="utf-8")
  handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
  logger.addHandler(handler)
 return logger
