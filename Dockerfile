FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies including libraries needed for OpenCV and dlib
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    curl \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p models api/temp api/output

# Copy source code
COPY . .

# Download shape predictor model if not exists
RUN mkdir -p models && \
    if [ ! -f models/shape_predictor_68_face_landmarks.dat ]; then \
    wget -q -O models/shape_predictor_68_face_landmarks.dat.bz2 https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2 && \
    bzip2 -d models/shape_predictor_68_face_landmarks.dat.bz2; \
    fi

# Expose the API port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models
ENV PYTHONUNBUFFERED=1

# Run the API server
CMD ["python", "api/server.py"]
