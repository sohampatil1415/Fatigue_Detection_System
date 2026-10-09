import os

class Config:
    """Central Application Configuration & Vision Thresholds."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'driver-drowsiness-detection-secret-key-2026')
    BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    
    # SQLite Database Config
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    DATABASE_PATH = os.path.join(DATA_DIR, 'driver_drowsiness.db')
    
    # Logging Configuration
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    LOG_FILE = os.path.join(LOG_DIR, 'app.log')
    
    # Computer Vision & Drowsiness Engine Config
    EAR_THRESHOLD = 0.25         # Eye Aspect Ratio threshold (Below this = closed eyes)
    EAR_CONSEC_FRAMES = 20       # Sustained frames below EAR_THRESHOLD before alarming (~1.0s to 1.5s)
    BLINK_EAR_THRESHOLD = 0.23   # EAR threshold to record a blink
    
    # Audio Alert Configuration
    AUDIO_ALARM_PATH = os.path.join(BASE_DIR, 'app', 'static', 'audio', 'alarm.wav')
    ALARM_ENABLED = True
