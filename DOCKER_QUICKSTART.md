# Wan2.1 Docker Quick Start

Get Wan2.1 running in Docker in 5 minutes!

## Prerequisites

- Docker 20.10+ installed ([Get Docker](https://docs.docker.com/get-docker/))
- NVIDIA GPU with 8GB+ VRAM (for GPU acceleration)
- NVIDIA Docker runtime installed ([Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

## Quick Start (3 Steps)

### Step 1: Clone and Navigate

```bash
git clone https://github.com/Wan-Video/Wan2.1.git
cd Wan2.1
```

### Step 2: Build and Start

**Option A: Using the helper script** (Recommended)

```bash
./docker-run.sh start
```

**Option B: Using Make**

```bash
make docker-build
make docker-up
```

**Option C: Using Docker Compose directly**

```bash
docker compose up -d wan2-1
```

### Step 3: Download Models and Run

```bash
# Enter the container
./docker-run.sh shell
# OR
make docker-shell
# OR
docker compose exec wan2-1 bash

# Download a model (1.3B for consumer GPUs)
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir /app/models/Wan2.1-T2V-1.3B

# Generate your first video!
python generate.py \
  --task t2v-1.3B \
  --size 832*480 \
  --ckpt_dir /app/models/Wan2.1-T2V-1.3B \
  --offload_model True \
  --t5_cpu \
  --sample_shift 8 \
  --sample_guide_scale 6 \
  --prompt "A cute cat playing with a ball of yarn"

# Your video will be in /app/outputs (accessible at ./outputs on your host)
```

## Common Commands

### Container Management

```bash
# Start container
./docker-run.sh start

# Stop container
./docker-run.sh stop

# Restart container
./docker-run.sh restart

# View logs
./docker-run.sh logs

# Enter shell
./docker-run.sh shell

# Check status
./docker-run.sh status
```

### Using Make Commands

```bash
make docker-up        # Start
make docker-down      # Stop
make docker-shell     # Enter shell
make docker-logs      # View logs
make docker-status    # Check status
make help            # Show all commands
```

## Run Gradio Web Interface

```bash
# Inside the container
cd gradio
python t2v_14B_singleGPU.py --ckpt_dir /app/models/Wan2.1-T2V-1.3B

# Open browser to: http://localhost:7860
```

## Available Models

| Model | VRAM | Resolution | Download Command |
|-------|------|------------|------------------|
| T2V-1.3B | 8GB+ | 480P | `huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir /app/models/Wan2.1-T2V-1.3B` |
| T2V-14B | 24GB+ | 720P | `huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir /app/models/Wan2.1-T2V-14B` |
| I2V-14B-720P | 24GB+ | 720P | `huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir /app/models/Wan2.1-I2V-14B-720P` |
| I2V-14B-480P | 16GB+ | 480P | `huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir /app/models/Wan2.1-I2V-14B-480P` |

## Troubleshooting

### "CUDA out of memory"
- Use the 1.3B model with `--offload_model True --t5_cpu`
- Reduce resolution to 480P

### "nvidia-smi not found"
- Ensure NVIDIA Docker runtime is installed
- Run: `docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`

### Can't access Gradio interface
- Check if port 7860 is exposed: `docker ps | grep 7860`
- Try: `http://127.0.0.1:7860` instead of `localhost`

## Next Steps

- Read the full [DOCKER_SETUP.md](DOCKER_SETUP.md) for advanced configuration
- Check the main [README.md](README.md) for model details
- Join the [Discord community](https://discord.gg/AKNgpMK4Yj)

## File Structure

```
Wan2.1/
├── models/          # Downloaded models (created automatically)
├── outputs/         # Generated videos (accessible from host)
├── cache/           # Model cache
├── Dockerfile       # Docker image definition
├── docker-compose.yml  # Container orchestration
├── docker-run.sh    # Helper script
├── Makefile         # Make commands
└── DOCKER_SETUP.md  # Detailed documentation
```

**Happy Generating!** 🎬
