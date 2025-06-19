import logging
import psutil
import os
import gc
from .config import MODEL_PATH, LANG_LIST, ENABLE_GPU
from .exceptions import ModelLoadError

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.model = None
        self.ocr_reader = None
        self.device = "cpu"  # Force CPU for Railway
        # Don't load models in __init__ to avoid import errors
        self._models_loaded = False
    
    def _load_models(self):
        """Load models with Railway compatibility"""
        if self._models_loaded:
            return
            
        try:
            # Import here to avoid early import issues
            from ultralytics import YOLO
            import easyocr
            
            logger.info(f"Loading YOLO model from {MODEL_PATH}")
            
            # Check if model file exists
            if not os.path.exists(MODEL_PATH):
                logger.error(f"Model file not found: {MODEL_PATH}")
                raise ModelLoadError(f"Model file not found: {MODEL_PATH}")
            
            # Load model with error handling
            self.model = YOLO(MODEL_PATH)
            self.model.to("cpu")
            
            # Force garbage collection
            gc.collect()
            
            logger.info("YOLO model loaded successfully on CPU")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise ModelLoadError(f"Failed to load YOLO model: {str(e)}")
        
        try:
            import easyocr
            logger.info(f"Loading OCR reader with languages: {LANG_LIST}")
            
            # Configure EasyOCR for Railway
            self.ocr_reader = easyocr.Reader(
                LANG_LIST,
                gpu=False,  # Force CPU
                verbose=False,
                download_enabled=True
            )
            
            # Force garbage collection
            gc.collect()
            
            logger.info("OCR reader loaded successfully")
            self._models_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load OCR reader: {e}")
            raise ModelLoadError(f"Failed to load OCR reader: {str(e)}")
    
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
        """Get current memory usage"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return {
                "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
                "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
                "percent": round(process.memory_percent(), 2)
            }
        except:
            return {"error": "Unable to get memory info"}

# Global model manager instance (lazy loading)
model_manager = ModelManager()