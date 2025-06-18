# app/predict.py
import numpy as np
from PIL import Image
import logging
import torch
from typing import List, Dict, Any, Tuple
from .model import model_manager
from .utils import preprocess_image, cleanup_memory
from .config import MIN_DETECTION_CONFIDENCE, MIN_OCR_CONFIDENCE
from .exceptions import APIException, ProcessingError

logger = logging.getLogger(__name__)

def clean_plate_text(text: str) -> str:
    """Clean and format detected plate text with improved logic"""
    if not text:
        return ""
    
    # Remove extra spaces and convert to uppercase
    cleaned = ' '.join(text.split()).upper()
    
    # Common OCR corrections for license plates
    replacements = {
        'O': '0',  # Letter O to number 0
        'I': '1',  # Letter I to number 1
        'S': '5',  # Letter S to number 5 (sometimes)
        'B': '8',  # Letter B to number 8 (sometimes)
    }
    
    # Apply replacements selectively
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    
    # Remove special characters except spaces and hyphens
    cleaned = ''.join(c for c in cleaned if c.isalnum() or c in ' -')
    
    return cleaned.strip()

def adjust_bbox_for_scale(bbox: List[int], scale_factor: float) -> List[int]:
    """Adjust bounding box coordinates back to original image scale"""
    if scale_factor == 1.0:
        return bbox
    
    return [int(coord / scale_factor) for coord in bbox]

def predict_plate(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    High-resolution optimized license plate prediction
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        List of detected plates with text and confidence
    """
    try:
        # Preprocess image with scale tracking
        image, scale_factor = preprocess_image(image_bytes)
        image_np = np.array(image)
        
        # Get models
        model = model_manager.get_model()
        ocr_reader = model_manager.get_ocr_reader()
        
        # Run YOLO detection optimized for high-resolution images
        logger.info(f"Running YOLO detection on {image_np.shape[1]}x{image_np.shape[0]} image")
        
        # Configure inference parameters for high-resolution processing
        with torch.inference_mode():
            results = model(
                image_np,
                conf=MIN_DETECTION_CONFIDENCE,
                iou=0.45,  # NMS IoU threshold
                max_det=20,  # Increased for high-res images that may have more plates
                verbose=False,
                imgsz=None  # Let YOLO handle image size automatically
            )
        
        plates = []
        
        # Process detections
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i, box in enumerate(boxes):
                try:
                    # Extract detection info
                    class_id = int(box.cls.item())
                    class_name = model.names[class_id]
                    confidence = float(box.conf.item())
                    
                    # Filter by class and confidence
                    if class_name.lower() not in ["plate", "license_plate", "number_plate"]:
                        continue
                    
                    if confidence < MIN_DETECTION_CONFIDENCE:
                        continue
                    
                    # Extract and validate bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # Ensure valid coordinates
                    x1 = max(0, min(x1, image_np.shape[1]))
                    y1 = max(0, min(y1, image_np.shape[0]))
                    x2 = max(x1 + 1, min(x2, image_np.shape[1]))
                    y2 = max(y1 + 1, min(y2, image_np.shape[0]))
                    
                    # Skip invalid boxes
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Crop plate region with adaptive padding based on image size
                    # Larger images get more padding for better OCR
                    padding = max(2, min(10, int(min(image_np.shape[:2]) * 0.01)))
                    crop_x1 = max(0, x1 - padding)
                    crop_y1 = max(0, y1 - padding)
                    crop_x2 = min(image_np.shape[1], x2 + padding)
                    crop_y2 = min(image_np.shape[0], y2 + padding)
                    
                    crop = image_np[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    if crop.size == 0:
                        continue
                    
                    # Enhance crop quality for better OCR on high-res images
                    crop_height, crop_width = crop.shape[:2]
                    
                    # If cropped region is too small, resize it for better OCR
                    if crop_height < 50 or crop_width < 150:
                        scale_up = max(2.0, 50 / crop_height, 150 / crop_width)
                        new_h, new_w = int(crop_height * scale_up), int(crop_width * scale_up)
                        crop_pil = Image.fromarray(crop)
                        crop_pil = crop_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        crop = np.array(crop_pil)
                    
                    # Run OCR with optimized parameters
                    logger.info(f"Running OCR on plate {i+1}")
                    try:
                        ocr_results = ocr_reader.readtext(
                            crop,
                            detail=True,
                            allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ',
                            width_ths=0.5,   # More lenient for high-res
                            height_ths=0.5,  # More lenient for high-res
                            paragraph=False,  # Process individual text segments
                            batch_size=1
                        )
                    except Exception as ocr_error:
                        logger.warning(f"OCR failed for detection {i}: {ocr_error}")
                        continue
                    
                    # Process OCR results
                    if ocr_results:
                        texts = []
                        confidences = []
                        
                        for (bbox_ocr, text, conf) in ocr_results:
                            if conf > MIN_OCR_CONFIDENCE and text.strip():
                                texts.append(text.strip())
                                confidences.append(conf)
                        
                        if texts:
                            combined_text = ' '.join(texts)
                            cleaned_text = clean_plate_text(combined_text)
                            
                            if cleaned_text:  # Only add if we have valid text
                                avg_ocr_confidence = sum(confidences) / len(confidences)
                                overall_confidence = (confidence + avg_ocr_confidence) / 2
                                
                                # Adjust bbox back to original scale
                                original_bbox = adjust_bbox_for_scale([x1, y1, x2, y2], scale_factor)
                                
                                plates.append({
                                    "text": cleaned_text,
                                    "confidence": round(overall_confidence, 3),
                                    "bbox": original_bbox,
                                    "detection_confidence": round(confidence, 3),
                                    "ocr_confidence": round(avg_ocr_confidence, 3)
                                })
                                
                                logger.info(f"Detected plate: {cleaned_text} (confidence: {overall_confidence:.3f})")
                
                except Exception as e:
                    logger.warning(f"Error processing detection {i}: {e}")
                    continue
        
        # Cleanup memory after processing
        cleanup_memory()
        
        logger.info(f"Total plates detected: {len(plates)}")
        return plates
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        # Cleanup on error
        cleanup_memory()
        raise ProcessingError(f"Prediction failed: {str(e)}")