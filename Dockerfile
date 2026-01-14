FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libv4l-dev \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir requests gunicorn

# Copy application code
COPY app.py .
COPY templates/ templates/
COPY static/ static/

# Create necessary directories
RUN mkdir -p Attendance static/faces

# Expose port (Railway will override with PORT env var)
EXPOSE 5000

# Run with gunicorn for production
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 app:app
