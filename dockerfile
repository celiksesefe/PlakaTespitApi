# Dockerfile
FROM python:3.11-slim

# Set environment variables for headless operation
ENV DEBIAN_FRONTEND=noninteractive
ENV QT_QPA_PLATFORM=offscreen
ENV MPLBACKEND=Agg
ENV DISPLAY=:99
ENV OPENCV_VIDEOIO_PRIORITY_MSMF=0
ENV OPENCV_DNN_OPENCL_ALLOW_ALL_DEVICES=0
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # OpenGL and graphics libraries
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    mesa-utils \
    # X11 and GUI libraries
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxi6 \
    # Font and text rendering
    libfontconfig1 \
    libfreetype6 \
    fonts-dejavu-core \
    # Other dependencies
    libgomp1 \
    libgtk-3-0 \
    libasound2 \
    # Virtual display for headless operation
    xvfb \
    # Utilities
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads logs

# Set proper permissions
RUN chmod +x run.py

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start virtual display in background and run the application
CMD ["sh", "-c", "Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset & python run.py"]