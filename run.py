import uvicorn
import os
import sys

def check_environment():
    """Check if required files exist"""
    print("=== Environment Check ===")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    
    required_files = ["app/main.py", "app/config.py", "requirements.txt"]
    for file in required_files:
        exists = os.path.exists(file)
        print(f"  {file}: {'✓' if exists else '✗'}")
        if not exists:
            print(f"    WARNING: {file} not found!")
    
    # Check for model file
    model_path = os.getenv("MODEL_PATH", "yolov8best.pt")
    model_exists = os.path.exists(model_path)
    print(f"  Model ({model_path}): {'✓' if model_exists else '✗'}")
    if not model_exists:
        print(f"    WARNING: Model file {model_path} not found!")
    
    print("========================\n")
    return model_exists

if __name__ == "__main__":
    # Check environment first
    model_exists = check_environment()
    
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"
    
    print(f"Starting server on {host}:{port}")
    
    if not model_exists:
        print("WARNING: Model file not found. API will fail on prediction requests.")
    
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