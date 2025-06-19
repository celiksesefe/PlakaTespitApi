import os
from pathlib import Path

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8best.pt")
LANG_LIST = ['en']

# API configuration - Docker optimized
MAX_FILE_SIZE = 30 * 1024 * 1024  # 30MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}

# Detection configuration
MIN_DETECTION_CONFIDENCE = 0.5
MIN_OCR_CONFIDENCE = 0.6

# Image processing configuration
MAX_PROCESSING_SIZE = 1920
MAX_IMAGE_SIZE = 4096
MIN_IMAGE_SIZE = 320
JPEG_QUALITY = 95
SMART_RESIZE = True

# Memory management - Docker optimized
MAX_BATCH_SIZE = 1
ENABLE_GPU = False  # Force CPU in Docker

# Paths
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)