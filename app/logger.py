import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(app):
    """Configures system-wide logging to console and app.log file."""
    log_dir = app.config['LOG_DIR']
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = app.config['LOG_FILE']
    
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s (%(pathname)s:%(lineno)d): %(message)s'
    )
    
    file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    
    app.logger.setLevel(logging.DEBUG)
    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)

    # Enable DEBUG for vision modules so EAR diagnostics appear
    for mod_name in ['app.vision.drowsiness_detector', 'app.vision.face_mesh']:
        mod_logger = logging.getLogger(mod_name)
        mod_logger.setLevel(logging.DEBUG)
        if not mod_logger.handlers:
            mod_logger.addHandler(console_handler)
    
    app.logger.info("System logger initialized successfully.")
