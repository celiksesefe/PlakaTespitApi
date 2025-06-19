import os
import sys
import uvicorn

# Set headless environment variables BEFORE importing anything else
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_DNN_OPENCL_ALLOW_ALL_DEVICES'] = '0'
os.environ['DISPLAY'] = ':99'  # Virtual display

def check_environment():
    """Check environment and files"""
    print("=== Environment Check ===")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    
    # Check key environment variables
    env_vars = ['QT_QPA_PLATFORM', 'MPLBACKEND', 'DISPLAY']
    for var in env_vars:
        value = os.getenv(var, 'Not set')
        print(f"  {var}: {value}")
    
    # Check required files
    required_files = ["app/main.py", "app/config.py", "requirements.txt"]
    for file in required_files:
        exists = os.path.exists(file)
        print(f"  {file}: {'✓' if exists else '✗'}")
    
    # Check model file
    model_path = os.getenv("MODEL_PATH", "yolov8best.pt")
    model_exists = os.path.exists(model_path)
    print(f"  Model ({model_path}): {'✓' if model_exists else '✗'}")
    
    print("========================\n")
    return model_exists

if __name__ == "__main__":
    model_exists = check_environment()
    
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"
    
    print(f"Starting server on {host}:{port}")
    
    if not model_exists:
        print("WARNING: Model file not found!")
    
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
        print(f"Failed to start server: {e}")
        import traceback
        traceback.print_exc()