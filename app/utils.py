# app/utils.py
import os
from pathlib import Path
from PIL import Image, ImageOps
import io
import gc
import logging
from typing import Tuple
from .config import (
    MAX_FILE_SIZE, ALLOWED_EXTENSIONS, MAX_IMAGE_SIZE,
    MIN_IMAGE_SIZE, JPEG_QUALITY
)
from .exceptions import InvalidImageError, FileSizeError

logger = logging.getLogger(__name__)

def validate_image(file_content: bytes, filename: str) -> None:
    """Validate uploaded high-resolution image file"""
    
    # Check file size - 30MB limit
    if len(file_content) > MAX_FILE_SIZE:
        raise FileSizeError(f"File size exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit")
    
    # Check file extension - support all common formats
    file_ext = Path(filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise InvalidImageError(f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    
    # Validate image integrity and dimensions
    try:
        image = Image.open(io.BytesIO(file_content))
        image.verify()  # Verify image integrity
        
        # Re-open for dimension check (verify() closes the image)
        image = Image.open(io.BytesIO(file_content))
        width, height = image.size
        
        # Check minimum dimensions (but no maximum limit)
        if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
            raise InvalidImageError(f"Image too small. Minimum size: {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}")
        
        # Log image info for monitoring
        logger.info(f"Validating image: {width}x{height}, {len(file_content)/(1024*1024):.1f}MB")
            
    except Exception as e:
        if isinstance(e, InvalidImageError):
            raise
        raise InvalidImageError("Invalid or corrupted image file")

def smart_resize_for_detection(image: Image.Image) -> Tuple[Image.Image, float]:
    """
    Smart resizing that maintains quality for license plate detection
    Only resize if absolutely necessary for memory constraints
    """
    original_width, original_height = image.size
    max_dimension = max(original_width, original_height)
    
    # Only resize if image is extremely large (>4K resolution)
    if max_dimension > MAX_IMAGE_SIZE:
        scale_factor = MAX_IMAGE_SIZE / max_dimension
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        # Use highest quality resampling
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logger.info(f"Resized image from {original_width}x{original_height} to {new_width}x{new_height}")
    else:
        scale_factor = 1.0
        logger.info(f"Processing at original resolution: {original_width}x{original_height}")
    
    return image, scale_factor

def optimize_image_size(image: Image.Image) -> Tuple[Image.Image, float]:
    """
    High-resolution optimized processing - minimal resizing
    """
    # Use smart resize instead of aggressive downsizing
    return smart_resize_for_detection(image)

def preprocess_image(image_bytes: bytes) -> Tuple[Image.Image, float]:
    """
    High-resolution image preprocessing with minimal quality loss
    """
    try:
        # Open and convert to RGB
        image = Image.open(io.BytesIO(image_bytes))
        
        # Handle EXIF orientation
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        
        # Smart resize - only if absolutely necessary
        image, scale_factor = optimize_image_size(image)
        
        # Enhance image quality for better license plate detection
        # Optional: Apply subtle contrast enhancement for better OCR
        if image.size[0] * image.size[1] < 2000000:  # Only for smaller images
            image = ImageOps.autocontrast(image, cutoff=1)
        
        return image, scale_factor
        
    except Exception as e:
        raise InvalidImageError(f"Failed to process image: {str(e)}")

def cleanup_memory():
    """Force garbage collection for server memory management"""
    gc.collect()
