# app/config.py
import os
from pathlib import Path

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8best.pt")
LANG_LIST = ['en']

# API configuration - High resolution support
MAX_FILE_SIZE = 30 * 1024 * 1024  # 30MB for high-resolution images
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}

# Detection configuration
MIN_DETECTION_CONFIDENCE = 0.5
MIN_OCR_CONFIDENCE = 0.6

# Image processing configuration - High resolution optimized
MAX_PROCESSING_SIZE = 1920  # Process at high resolution for better detection
MAX_IMAGE_SIZE = 4096       # Only resize if larger than 4K
MIN_IMAGE_SIZE = 320
JPEG_QUALITY = 95           # High quality for better OCR
SMART_RESIZE = True         # Enable smart resizing based on content

# Memory management
MAX_BATCH_SIZE = 1
ENABLE_GPU = os.getenv("ENABLE_GPU", "false").lower() == "true"

# Paths
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)