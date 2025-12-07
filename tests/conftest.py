"""
Pytest configuration and shared fixtures for Wan2.1 tests.

Copyright (c) 2025 Kuaishou. All rights reserved.
"""

import pytest
import torch
import tempfile
from pathlib import Path
from typing import Dict, Any


@pytest.fixture(scope="session")
def device():
    """Return the device to use for testing (CPU or CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def dtype():
    """Return the default dtype for testing."""
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_14b() -> Dict[str, Any]:
    """Return a minimal 14B model configuration for testing."""
    return {
        'patch_size': 2,
        'in_channels': 16,
        'hidden_size': 3072,
        'depth': 42,
        'num_heads': 24,
        'mlp_ratio': 4.0,
        'learn_sigma': True,
        'qk_norm': True,
        'qk_norm_type': 'rms',
        'norm_type': 'rms',
        'posemb_type': 'rope2d_video',
        'num_experts': 1,
        'route_method': 'soft',
        'router_top_k': 1,
        'pooled_projection_type': 'linear',
        'cap_feat_dim': 4096,
        'caption_channels': 4096,
        't5_feat_dim': 2048,
        'text_len': 512,
        'use_attention_mask': True,
    }


@pytest.fixture
def sample_config_1_3b() -> Dict[str, Any]:
    """Return a minimal 1.3B model configuration for testing."""
    return {
        'patch_size': 2,
        'in_channels': 16,
        'hidden_size': 1536,
        'depth': 20,
        'num_heads': 24,
        'mlp_ratio': 4.0,
        'learn_sigma': True,
        'qk_norm': True,
        'qk_norm_type': 'rms',
        'norm_type': 'rms',
        'posemb_type': 'rope2d_video',
        'num_experts': 1,
        'route_method': 'soft',
        'router_top_k': 1,
        'pooled_projection_type': 'linear',
        'cap_feat_dim': 4096,
        'caption_channels': 4096,
        't5_feat_dim': 2048,
        'text_len': 512,
        'use_attention_mask': True,
    }


@pytest.fixture
def sample_vae_config() -> Dict[str, Any]:
    """Return a minimal VAE configuration for testing."""
    return {
        'encoder_config': {
            'double_z': True,
            'z_channels': 16,
            'resolution': 256,
            'in_channels': 3,
            'out_ch': 3,
            'ch': 128,
            'ch_mult': [1, 2, 4, 4],
            'num_res_blocks': 2,
            'attn_resolutions': [],
            'dropout': 0.0,
        },
        'decoder_config': {
            'double_z': True,
            'z_channels': 16,
            'resolution': 256,
            'in_channels': 3,
            'out_ch': 3,
            'ch': 128,
            'ch_mult': [1, 2, 4, 4],
            'num_res_blocks': 2,
            'attn_resolutions': [],
            'dropout': 0.0,
        },
        'temporal_compress_level': 4,
    }


@pytest.fixture
def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture
def skip_if_no_flash_attn():
    """Skip test if flash_attn is not available."""
    try:
        import flash_attn
    except ImportError:
        pytest.skip("flash_attn not available")
