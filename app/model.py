import os
import logging
import psutil
import gc
import torch

# Set environment variables for headless operation
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLBACKEND', 'Agg')

from .config import MODEL_PATH, LANG_LIST, ENABLE_GPU
from .exceptions import ModelLoadError

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.model = None
        self.ocr_reader = None
        self.device = "cpu"
        self._models_loaded = False
        logger.info("ModelManager initialized")
    
    def _load_models(self):
        """Load models in Docker environment"""
        if self._models_loaded:
            return
            
        logger.info("Starting model loading process...")
        
        try:
            # Set PyTorch to use single thread for stability
            torch.set_num_threads(1)
            
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model from {MODEL_PATH}")
            
            # Check if model file exists
            if not os.path.exists(MODEL_PATH):
                logger.error(f"Model file not found: {MODEL_PATH}")
                # List files in current directory for debugging
                files = os.listdir(".")
                logger.info(f"Files in current directory: {files}")
                raise ModelLoadError(f"Model file not found: {MODEL_PATH}")
            
            # Load YOLO model
            self.model = YOLO(MODEL_PATH)
            self.model.to("cpu")
            
            # Configure for headless operation
            if hasattr(self.model, 'overrides'):
                self.model.overrides.update({
                    'verbose': False,
                    'plots': False,
                    'save': False,
                    'save_txt': False
                })
            
            logger.info("✓ YOLO model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise ModelLoadError(f"Failed to load YOLO model: {str(e)}")
        
        try:
            import easyocr
            logger.info(f"Loading EasyOCR with languages: {LANG_LIST}")
            
            # Load EasyOCR
            self.ocr_reader = easyocr.Reader(
                LANG_LIST,
                gpu=False,
                verbose=False,
                download_enabled=True
            )
            
            logger.info("✓ EasyOCR loaded successfully")
            self._models_loaded = True
            
            # Force garbage collection
            gc.collect()
            
            logger.info("✓ All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load EasyOCR: {e}")
            raise ModelLoadError(f"Failed to load EasyOCR: {str(e)}")
    
    def get_model(self):
        if not self._models_loaded:
            self._load_models()
        if self.model is None:
            raise ModelLoadError("YOLO model not loaded")
        return self.model
    
    def get_ocr_reader(self):
        if not self._models_loaded:
            self._load_models()
        if self.ocr_reader is None:
            raise ModelLoadError("OCR reader not loaded")
        return self.ocr_reader
    
    def get_memory_usage(self) -> dict:
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return {
                "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
                "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
                "percent": round(process.memory_percent(), 2)
            }
        except Exception as e:
            logger.warning(f"Could not get memory info: {e}")
            return {"error": "Unable to get memory info"}

# Global model manager
model_manager = ModelManager()