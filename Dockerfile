# Dockerfile - use Python 3.12, install system deps for OpenCV / faiss
FROM python:3.12-slim

# Install system packages required by opencv/faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libglib2.0-0 libsm6 libxrender1 libxext6 \
    wget unzip git pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Use pip without cache to keep image small
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
ENV PORT 5000
# Use gunicorn to serve app; ensure scalable_attendance_system exposes `app`
CMD ["gunicorn", "-b", "0.0.0.0:5000", "scalable_attendance_system:app"]
