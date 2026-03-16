# Base Image with CUDA support
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirement files first for better caching
COPY pyproject.toml .
COPY README.md .

# Install dependencies
RUN pip3 install --no-cache-dir .

# Install API dependencies
RUN pip3 install uvicorn fastapi python-multipart

# Copy the rest of the application
COPY . .

# Set Environment Variables
ENV PYTHONPATH=/app
ENV AUTOTRANDHD_MODEL=/app/artifacts/checkpoints/best.pt
ENV AUTOTRANDHD_DEVICE=cuda

# Expose API port
EXPOSE 8000

# Run API by default
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
