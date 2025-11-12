# Dockerfile - forces Python 3.12 and installs system deps for OpenCV/faiss
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    wget unzip git pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy code
COPY . /app

# Upgrade pip and install requirements
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
ENV PORT 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "scalable_attendance_system:app"]
