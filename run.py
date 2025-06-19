import uvicorn
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check Docker environment"""
    logger.info("=== Docker Environment Check ===")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Python version: {sys.version}")
    
    # Check environment variables
    env_vars = ['QT_QPA_PLATFORM', 'MPLBACKEND', 'DISPLAY', 'PORT']
    for var in env_vars:
        value = os.getenv(var, 'Not set')
        logger.info(f"  {var}: {value}")
    
    # Check required files
    required_files = ["app/main.py", "app/config.py"]
    for file in required_files:
        exists = os.path.exists(file)
        logger.info(f"  {file}: {'✓' if exists else '✗'}")
    
    # Check model file
    model_path = os.getenv("MODEL_PATH", "yolov8best.pt")
    model_exists = os.path.exists(model_path)
    logger.info(f"  Model ({model_path}): {'✓' if model_exists else '✗'}")
    
    logger.info("================================")
    return model_exists

if __name__ == "__main__":
    # Check environment
    model_exists = check_environment()
    
    # Get port from environment (Railway sets this)
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"
    
    logger.info(f"Starting server on {host}:{port}")
    
    if not model_exists:
        logger.warning("Model file not found. Predictions will fail.")
    
    try:
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            workers=1,
            log_level="info",
            access_log=True,
            reload=False
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)