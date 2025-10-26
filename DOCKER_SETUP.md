# Wan2.1 Docker Setup Guide

Professional-grade instructions for running Wan2.1 video generation models in Docker containers with GPU support.

---

## Table of Contents
- [Prerequisites](#prerequisites)
- [System Requirements](#system-requirements)
- [Installation Steps](#installation-steps)
- [Quick Start](#quick-start)
- [Model Download](#model-download)
- [Running Inference](#running-inference)
- [Gradio Web Interface](#gradio-web-interface)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)

---

## Prerequisites

### Required Software

1. **Docker Engine** (version 20.10+)
   - [Installation Guide](https://docs.docker.com/engine/install/)

2. **NVIDIA Docker Runtime** (for GPU support)
   - Required for GPU acceleration
   - [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

3. **NVIDIA Drivers** (version 525.60.13+)
   - CUDA 12.1 compatible drivers
   - Check with: `nvidia-smi`

4. **Docker Compose** (version 2.0+)
   - Typically included with Docker Desktop
   - [Installation Guide](https://docs.docker.com/compose/install/)

### Optional Software

- **Git** - For cloning the repository
- **Make** - For using convenience commands
- **NVIDIA Container Toolkit** - For multi-GPU support

---

## System Requirements

### Minimum Requirements (T2V-1.3B at 480P)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (e.g., RTX 4060 Ti)
- **RAM**: 16GB system memory
- **Storage**: 50GB free space (for models and cache)
- **OS**: Linux (Ubuntu 20.04+), Windows 10/11 with WSL2

### Recommended Requirements (T2V-14B at 720P)
- **GPU**: NVIDIA GPU with 24GB+ VRAM (e.g., RTX 4090, A5000)
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ free space
- **OS**: Linux (Ubuntu 22.04+)

### Multi-GPU Setup (for 8x GPU)
- **GPUs**: 8x NVIDIA GPUs (A100, H100, etc.)
- **RAM**: 128GB+ system memory
- **Storage**: 200GB+ free space
- **Network**: High-bandwidth GPU interconnect (NVLink preferred)

---

## Installation Steps

### Step 1: Verify Docker and NVIDIA Runtime

```bash
# Check Docker installation
docker --version
docker compose version

# Check NVIDIA driver
nvidia-smi

# Test NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

**Expected output**: You should see your GPU(s) listed in the nvidia-smi output.

### Step 2: Clone the Repository

```bash
git clone https://github.com/Wan-Video/Wan2.1.git
cd Wan2.1
```

### Step 3: Create Required Directories

```bash
# Create directories for models, outputs, and cache
mkdir -p models outputs cache examples
```

### Step 4: Set Environment Variables (Optional)

For prompt extension with Dashscope API:

```bash
# Create a .env file
cat > .env << EOF
DASH_API_KEY=your_dashscope_api_key_here
DASH_API_URL=https://dashscope.aliyuncs.com/api/v1
EOF
```

For international Alibaba Cloud users:
```bash
DASH_API_URL=https://dashscope-intl.aliyuncs.com/api/v1
```

### Step 5: Build the Docker Image

```bash
# Build using Docker Compose (recommended)
docker compose build

# OR build manually
docker build -t wan2.1:latest .
```

**Build time**: Approximately 10-20 minutes depending on your internet connection.

---

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Start the container with GPU support
docker compose up -d wan2-1

# Check container status
docker compose ps

# View logs
docker compose logs -f wan2-1

# Access the container shell
docker compose exec wan2-1 bash
```

### Option 2: Using Docker Run

```bash
docker run -it --gpus all \
  --name wan2.1-container \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/cache:/app/cache \
  -p 7860:7860 \
  --shm-size=16g \
  wan2.1:latest bash
```

### For CPU-only Mode

```bash
# Using Docker Compose
docker compose --profile cpu up -d wan2-1-cpu

# Using Docker Run
docker run -it \
  --name wan2.1-cpu \
  -e CUDA_VISIBLE_DEVICES="" \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/cache:/app/cache \
  -p 7860:7860 \
  wan2.1:latest bash
```

---

## Model Download

Download models **before** running inference. Models should be placed in the `./models` directory.

### Using Hugging Face CLI (Inside Container)

```bash
# Enter the container
docker compose exec wan2-1 bash

# Download T2V-14B model
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir /app/models/Wan2.1-T2V-14B

# Download T2V-1.3B model
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir /app/models/Wan2.1-T2V-1.3B

# Download I2V-14B-720P model
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir /app/models/Wan2.1-I2V-14B-720P

# Download I2V-14B-480P model
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir /app/models/Wan2.1-I2V-14B-480P

# Download FLF2V-14B model
huggingface-cli download Wan-AI/Wan2.1-FLF2V-14B-720P --local-dir /app/models/Wan2.1-FLF2V-14B-720P

# Download VACE models
huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B --local-dir /app/models/Wan2.1-VACE-1.3B
huggingface-cli download Wan-AI/Wan2.1-VACE-14B --local-dir /app/models/Wan2.1-VACE-14B
```

### Using ModelScope (Alternative for Chinese Users)

```bash
pip install modelscope
modelscope download Wan-AI/Wan2.1-T2V-14B --local_dir /app/models/Wan2.1-T2V-14B
```

### Download from Host Machine

You can also download models on your host machine and they will be accessible in the container:

```bash
# On host machine (outside Docker)
cd Wan2.1/models
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
```

---

## Running Inference

All commands below should be run **inside the container**.

### Text-to-Video Generation

#### 1.3B Model (480P) - Consumer GPU Friendly

```bash
python generate.py \
  --task t2v-1.3B \
  --size 832*480 \
  --ckpt_dir /app/models/Wan2.1-T2V-1.3B \
  --offload_model True \
  --t5_cpu \
  --sample_shift 8 \
  --sample_guide_scale 6 \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

#### 14B Model (720P) - High-End GPU

```bash
python generate.py \
  --task t2v-14B \
  --size 1280*720 \
  --ckpt_dir /app/models/Wan2.1-T2V-14B \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

#### With Prompt Extension (Better Quality)

```bash
# Using local Qwen model
python generate.py \
  --task t2v-14B \
  --size 1280*720 \
  --ckpt_dir /app/models/Wan2.1-T2V-14B \
  --use_prompt_extend \
  --prompt_extend_method 'local_qwen' \
  --prompt "A beautiful sunset over the ocean"

# Using Dashscope API (requires DASH_API_KEY)
DASH_API_KEY=your_key python generate.py \
  --task t2v-14B \
  --size 1280*720 \
  --ckpt_dir /app/models/Wan2.1-T2V-14B \
  --use_prompt_extend \
  --prompt_extend_method 'dashscope' \
  --prompt "A beautiful sunset over the ocean"
```

### Image-to-Video Generation

```bash
python generate.py \
  --task i2v-14B \
  --size 1280*720 \
  --ckpt_dir /app/models/Wan2.1-I2V-14B-720P \
  --image /app/examples/i2v_input.JPG \
  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."
```

### First-Last-Frame-to-Video

```bash
python generate.py \
  --task flf2v-14B \
  --size 1280*720 \
  --ckpt_dir /app/models/Wan2.1-FLF2V-14B-720P \
  --first_frame /app/examples/flf2v_input_first_frame.png \
  --last_frame /app/examples/flf2v_input_last_frame.png \
  --prompt "CG animation style, a small blue bird takes off from the ground"
```

### Text-to-Image Generation

```bash
python generate.py \
  --task t2i-14B \
  --size 1024*1024 \
  --ckpt_dir /app/models/Wan2.1-T2V-14B \
  --prompt "A serene mountain landscape at dawn"
```

### VACE (Video Creation and Editing)

```bash
python generate.py \
  --task vace-1.3B \
  --size 832*480 \
  --ckpt_dir /app/models/Wan2.1-VACE-1.3B \
  --src_ref_images /app/examples/girl.png,/app/examples/snake.png \
  --prompt "Your detailed prompt here"
```

---

## Gradio Web Interface

### Start Gradio Interface

#### Text-to-Video (14B)

```bash
cd gradio
python t2v_14B_singleGPU.py \
  --ckpt_dir /app/models/Wan2.1-T2V-14B \
  --prompt_extend_method 'local_qwen'
```

#### Image-to-Video (14B)

```bash
cd gradio
python i2v_14B_singleGPU.py \
  --ckpt_dir_720p /app/models/Wan2.1-I2V-14B-720P \
  --prompt_extend_method 'local_qwen'
```

#### VACE (All-in-One)

```bash
cd gradio
python vace.py --ckpt_dir /app/models/Wan2.1-VACE-1.3B
```

### Access the Web Interface

1. Open your web browser
2. Navigate to: `http://localhost:7860`
3. Use the intuitive interface to generate videos

### For Remote Access

If running on a remote server:

```bash
# Start with public URL (Gradio share feature)
python gradio/t2v_14B_singleGPU.py \
  --ckpt_dir /app/models/Wan2.1-T2V-14B \
  --server_name 0.0.0.0 \
  --server_port 7860 \
  --share
```

Then access via: `http://your-server-ip:7860`

---

## Advanced Configuration

### Multi-GPU Inference (FSDP + xDiT)

For 8-GPU setup using Ulysses or Ring attention strategies:

```bash
# Install xDiT
pip install "xfuser>=0.4.1"

# Run with Ulysses strategy (8 GPUs)
torchrun --nproc_per_node=8 generate.py \
  --task t2v-14B \
  --size 1280*720 \
  --ckpt_dir /app/models/Wan2.1-T2V-14B \
  --dit_fsdp \
  --t5_fsdp \
  --ulysses_size 8 \
  --prompt "Your prompt here"

# Run with Ring strategy (for sequence parallelism)
torchrun --nproc_per_node=8 generate.py \
  --task t2v-14B \
  --size 1280*720 \
  --ckpt_dir /app/models/Wan2.1-T2V-14B \
  --dit_fsdp \
  --t5_fsdp \
  --ring_size 8 \
  --prompt "Your prompt here"
```

### Memory Optimization Flags

For limited VRAM:

```bash
python generate.py \
  --task t2v-1.3B \
  --size 832*480 \
  --ckpt_dir /app/models/Wan2.1-T2V-1.3B \
  --offload_model True \  # Offload model to CPU when not in use
  --t5_cpu \               # Keep T5 encoder on CPU
  --sample_shift 8 \
  --sample_guide_scale 6 \
  --prompt "Your prompt"
```

### Custom Output Directory

```bash
python generate.py \
  --task t2v-14B \
  --size 1280*720 \
  --ckpt_dir /app/models/Wan2.1-T2V-14B \
  --output_dir /app/outputs/my_generation \
  --prompt "Your prompt"
```

### Batch Generation

Generate multiple variations:

```bash
python generate.py \
  --task t2v-14B \
  --size 1280*720 \
  --ckpt_dir /app/models/Wan2.1-T2V-14B \
  --base_seed 0 \
  --num_samples 4 \  # Generate 4 variations
  --prompt "Your prompt"
```

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solutions:**
1. Use smaller model (1.3B instead of 14B)
2. Reduce resolution (480P instead of 720P)
3. Enable memory optimization flags:
   ```bash
   --offload_model True --t5_cpu
   ```
4. Increase Docker shared memory:
   ```bash
   docker run --shm-size=32g ...
   ```

### Issue: "nvidia-smi not found" inside container

**Solutions:**
1. Verify NVIDIA Docker runtime is installed on host
2. Check Docker daemon configuration:
   ```bash
   # Edit /etc/docker/daemon.json
   {
     "runtimes": {
       "nvidia": {
         "path": "nvidia-container-runtime",
         "runtimeArgs": []
       }
     },
     "default-runtime": "nvidia"
   }
   ```
3. Restart Docker daemon:
   ```bash
   sudo systemctl restart docker
   ```

### Issue: "Flash attention installation failed"

**Solution:**
Flash attention is optional. The Dockerfile continues even if it fails. For better performance, install manually:

```bash
# Inside container
pip install flash-attn --no-build-isolation
```

### Issue: Model download fails

**Solutions:**
1. Check internet connection
2. Use mirror sites (ModelScope for Chinese users)
3. Download models on host machine and mount them
4. Increase Docker download timeout

### Issue: "RuntimeError: CUDA error: device-side assert triggered"

**Solutions:**
1. Check CUDA compatibility:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
2. Update NVIDIA drivers
3. Rebuild Docker image with matching CUDA version

### Issue: Gradio interface not accessible

**Solutions:**
1. Check if port is exposed:
   ```bash
   docker ps | grep 7860
   ```
2. Ensure firewall allows port 7860
3. Try binding to all interfaces:
   ```bash
   python gradio/app.py --server_name 0.0.0.0
   ```

### Issue: Permission denied errors

**Solution:**
```bash
# Fix ownership of mounted volumes
sudo chown -R $(id -u):$(id -g) models outputs cache
```

---

## Performance Optimization

### 1. Use SSD Storage
- Store models and cache on SSD for faster loading
- Use NVMe for best performance

### 2. Increase Shared Memory
```bash
# In docker-compose.yml
shm_size: '32gb'
```

### 3. Use Mixed Precision
- The model uses bfloat16 by default (optimal for modern GPUs)

### 4. Enable Xformers (if available)
```bash
pip install xformers
```

### 5. Multi-GPU Best Practices
- Use NVLink/NVSwitch for GPU communication
- Balance model sharding with Ulysses + Ring strategies
- Monitor GPU utilization: `watch -n 1 nvidia-smi`

### 6. Optimize Inference Parameters
```bash
# For T2V-1.3B
--sample_shift 8 \        # Adjust 8-12 based on quality
--sample_guide_scale 6    # Lower = faster, higher = better quality

# For T2V-14B
--sample_guide_scale 5.0  # Default recommended
```

### 7. Use Persistent Cache
```bash
# Models and transformers will be cached in ./cache
# Reusing the cache speeds up subsequent runs
```

---

## Container Management

### Stop Container
```bash
docker compose down
```

### Restart Container
```bash
docker compose restart wan2-1
```

### View Logs
```bash
docker compose logs -f wan2-1
```

### Clean Up
```bash
# Remove containers
docker compose down -v

# Remove images
docker rmi wan2.1:latest

# Clean up Docker system
docker system prune -a
```

### Update Container
```bash
# Pull latest code
git pull origin main

# Rebuild image
docker compose build --no-cache

# Restart containers
docker compose up -d
```

---

## Security Best Practices

1. **Do not commit API keys** to version control
2. **Use .env files** for sensitive environment variables
3. **Limit container privileges**: Avoid running as root
4. **Keep Docker updated** for security patches
5. **Scan images** for vulnerabilities:
   ```bash
   docker scan wan2.1:latest
   ```

---

## Support and Resources

- **GitHub Issues**: [https://github.com/Wan-Video/Wan2.1/issues](https://github.com/Wan-Video/Wan2.1/issues)
- **Discord**: [Join the community](https://discord.gg/AKNgpMK4Yj)
- **Technical Report**: [arXiv:2503.20314](https://arxiv.org/abs/2503.20314)
- **Docker Documentation**: [https://docs.docker.com/](https://docs.docker.com/)
- **NVIDIA Container Toolkit**: [https://github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

---

## License

This Docker setup follows the same Apache 2.0 License as the Wan2.1 project. See [LICENSE.txt](LICENSE.txt) for details.

---

**Last Updated**: 2025-10-26
**Version**: 1.0.0
**Maintainer**: Wan2.1 Community
