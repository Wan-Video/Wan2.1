# Wan2.1 Docker Image
# Professional-grade Docker setup for Wan2.1 video generation models
# Supports both CPU and GPU (NVIDIA CUDA) environments

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Copy requirements first for better layer caching
COPY requirements.txt /app/requirements.txt

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Install flash-attention separately (can be problematic)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || echo "Flash attention installation failed, continuing..."

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Create directories for models and outputs
RUN mkdir -p /app/models /app/outputs /app/cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/cache
ENV HF_HOME=/app/cache/huggingface
ENV TRANSFORMERS_CACHE=/app/cache/transformers
ENV CUDA_VISIBLE_DEVICES=0

# Expose port for Gradio
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('CUDA available:', torch.cuda.is_available())" || exit 1

# Default command (can be overridden)
CMD ["/bin/bash"]
