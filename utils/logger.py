"""
Sistema de logging para APU
"""
import logging
import sys
from pathlib import Path
from config.settings import LOGGING_CONFIG, BASE_DIR

def setup_logger(name: str = "APU") -> logging.Logger:
    """
    Configura y retorna un logger
    
    Args:
        name: Nombre del logger
        
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    
    # Si ya está configurado, retornarlo
    if logger.handlers:
        return logger
    
    logger.setLevel(LOGGING_CONFIG["level"])
    
    # Formatter
    formatter = logging.Formatter(LOGGING_CONFIG["format"])
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = BASE_DIR / LOGGING_CONFIG["file"]
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Logger global
logger = setup_logger()