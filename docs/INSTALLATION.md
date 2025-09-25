# Installation Guide

This guide covers installation for different GPU generations and operating systems.

## Requirements

- Python 3.10.9
- Conda or Python venv
- Compatible GPU (RTX 10XX or newer recommended)

## Installation for RTX 10XX to RTX 50XX (Stable)

This installation uses PyTorch 2.7.0 which is well-tested and stable.

### Step 1: Download and Setup Environment

```shell
# Clone the repository
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP

# Create Python 3.10.9 environment using conda
conda create -n wan2gp python=3.10.9
conda activate wan2gp
```

### Step 2: Install PyTorch

```shell
# Install PyTorch 2.7.0 with CUDA 12.8
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
```

### Step 3: Install Dependencies

```shell
# Install core dependencies
pip install -r requirements.txt
```

### Step 4: Optional Performance Optimizations

#### Sage Attention (30% faster), don't install with RTX 50xx as it is not compatible

```shell
# Windows only: Install Triton
pip install triton-windows 

# For both Windows and Linux
pip install sageattention==1.0.6 
```

#### Sage 2 Attention (40% faster)

```shell
# Windows
pip install triton-windows 
pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.1.1-windows/sageattention-2.1.1+cu126torch2.6.0-cp310-cp310-win_amd64.whl

# Linux (manual compilation required)
python -m pip install "setuptools<=75.8.2" --force-reinstall
git clone https://github.com/thu-ml/SageAttention
cd SageAttention 
pip install -e .
```

#### Flash Attention

```shell
# May require CUDA kernel compilation on Windows
pip install flash-attn==2.7.2.post1
```

 
## Attention Modes

WanGP supports several attention implementations:

- **SDPA** (default): Available by default with PyTorch
- **Sage**: 30% speed boost with small quality cost
- **Sage2**: 40% speed boost 
- **Flash**: Good performance, may be complex to install on Windows

### Attention GPU Compatibility

- RTX 10XX, 20XX: SDPA
- RTX 30XX, 40XX: SDPA, Flash Attention, Xformers, Sage, Sage2
- RTX 50XX: SDPA, SDPA, Flash Attention, Xformers, Sage2

## Performance Profiles

Choose a profile based on your hardware:

- **Profile 3 (LowRAM_HighVRAM)**: Loads entire model in VRAM, requires 24GB VRAM for 8-bit quantized 14B model
- **Profile 4 (LowRAM_LowVRAM)**: Default, loads model parts as needed, slower but lower VRAM requirement

## Troubleshooting

### Sage Attention Issues

If Sage attention doesn't work:

1. Check if Triton is properly installed
2. Clear Triton cache
3. Fallback to SDPA attention:
   ```bash
   python wgp.py --attention sdpa
   ```

### Memory Issues

- Use lower resolution or shorter videos
- Enable quantization (default)
- Use Profile 4 for lower VRAM usage
- Consider using 1.3B models instead of 14B models


For more troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md) 
