# Use Python 3.11 slim (recommended for compatibility)
FROM python:3.11-slim

# Install system packages needed for OpenCV / FAISS / pycairo / build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libcairo2-dev \
    libpango1.0-dev \
    libgirepository1.0-dev \
    libglib2.0-0 libglib2.0-dev \
    libsm6 libxrender1 libxext6 \
    libjpeg-dev \
    wget unzip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Upgrade pip/setuptools and install Python deps without caching wheel files in image
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
ENV PORT 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "scalable_attendance_system:app"]
