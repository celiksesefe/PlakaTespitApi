from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import time
import psutil
import os
from .predict import predict_plate
from .utils import validate_image
from .exceptions import APIException
from .model import model_manager

# Configure logging for server deployment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="License Plate Detection API",
    description="High-resolution optimized API for detecting and reading license plates from images",
    version="2.0.0"
)

# Middleware for large file uploads
class LargeUploadMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Set larger limits for file uploads
        request.scope["client_max_size"] = 30 * 1024 * 1024  # 30MB
        response = await call_next(request)
        return response

app.add_middleware(LargeUploadMiddleware)

# FastAPI configuration for large file uploads
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure specific origins in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Exception handler for custom API exceptions
@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "status_code": exc.status_code}
    )

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "License Plate Detection API is running",
        "status": "healthy",
        "version": "2.0.0",
        "max_file_size": "30MB",
        "supported_formats": [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]
    }

@app.get("/health")
async def health_check():
    """Detailed health check with system info"""
    memory_info = model_manager.get_memory_usage()
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "license-plate-detection-api",
        "version": "2.0.0",
        "memory_usage": memory_info,
        "models_loaded": {
            "yolo": model_manager.model is not None,
            "ocr": model_manager.ocr_reader is not None
        },
        "max_file_size": "30MB"
    }

@app.get("/system-info")
async def system_info():
    """System information endpoint for monitoring"""
    memory_info = model_manager.get_memory_usage()
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory": memory_info,
        "device": model_manager.device,
        "models_loaded": {
            "yolo": model_manager.model is not None,
            "ocr": model_manager.ocr_reader is not None
        }
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    High-resolution license plate prediction endpoint
    Supports up to 30MB files in various formats
    
    Args:
        file: Image file (jpg, png, webp, bmp, tiff)
        
    Returns:
        JSON with detected plates and their information
    """
    start_time = time.time()
    
    try:
        # Validate file
        if not file.filename:
            raise APIException("No file provided", 400)
        
        # Read file content
        image_bytes = await file.read()
        
        # Validate image
        validate_image(image_bytes, file.filename)
        
        # Predict plates
        plates = predict_plate(image_bytes)
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Processed {file.filename} ({len(image_bytes)/(1024*1024):.1f}MB) "
            f"in {processing_time:.2f}s, found {len(plates)} plates"
        )
        
        return {
            "success": True,
            "plates": plates,
            "count": len(plates),
            "processing_time": round(processing_time, 3),
            "filename": file.filename,
            "file_size_mb": round(len(image_bytes) / (1024 * 1024), 2),
            "api_version": "2.0.0"
        }
        
    except APIException:
        raise  # Re-raise API exceptions
    except Exception as e:
        logger.error(f"Unexpected error processing {file.filename}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )