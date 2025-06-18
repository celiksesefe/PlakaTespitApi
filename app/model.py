import torch
from ultralytics import YOLO
import easyocr
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
        self._load_models()
    
    def _load_models(self):
        """Load models with Railway memory constraints"""
        try:
            logger.info(f"Loading YOLO model from {MODEL_PATH}")
            
            # Load model with memory optimization
            self.model = YOLO(MODEL_PATH)
            self.model.to("cpu")
            
            # Force garbage collection
            gc.collect()
            
            logger.info("YOLO model loaded successfully on CPU")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise ModelLoadError(f"Failed to load YOLO model: {str(e)}")
        
        try:
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
            
        except Exception as e:
            logger.error(f"Failed to load OCR reader: {e}")
            raise ModelLoadError(f"Failed to load OCR reader: {str(e)}")
    
    def get_model(self):
        if self.model is None:
            raise ModelLoadError("YOLO model not loaded")
        return self.model
    
    def get_ocr_reader(self):
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

# Global model manager instance
model_manager = ModelManager()