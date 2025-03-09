"""
Logging Utilities
===============
Logging configuration and utilities.
"""

import os
import logging
import sys

def setup_logging(log_level=logging.INFO, log_dir="logs", log_file="traffic_rl.log"):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level for console output
        log_dir: Directory for log files
        log_file: Log file name
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger("TrafficRL")
    logger.setLevel(logging.DEBUG)  # Set root level to lowest to capture everything
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create handlers
    try:
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # File handler
        file_path = os.path.join(log_dir, log_file)
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)  # Always use DEBUG for file logging
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # Create formatter and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.info(f"Logging initialized. Log file: {file_path}")
        
    except Exception as e:
        # Fallback to console-only logging if file logging fails
        print(f"Warning: Could not set up file logging: {e}")
        
        # Ensure we have at least console logging
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        logger.warning(f"File logging disabled. Using console logging only.")
    
    return logger


def enable_debug_logging(logger=None):
    """
    Enable debug logging level for all handlers.
    
    Args:
        logger: Logger instance (uses root logger if None)
    """
    if logger is None:
        logger = logging.getLogger("TrafficRL")
    
    logger.setLevel(logging.DEBUG)
    
    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG)
    
    logger.debug("Debug logging enabled")