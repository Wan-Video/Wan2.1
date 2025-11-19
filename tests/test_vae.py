"""
Unit tests for WanVAE in Wan2.1.

Copyright (c) 2025 Kuaishou. All rights reserved.
"""

import pytest
import torch
from wan.modules.vae import WanVAE_


class TestWanVAE:
    """Test suite for WanVAE (3D Causal VAE)."""

    def test_vae_initialization(self, sample_vae_config):
        """Test VAE initialization."""
        with torch.device('meta'):
            vae = WanVAE_(**sample_vae_config)

        assert vae is not None
        assert hasattr(vae, 'encoder')
        assert hasattr(vae, 'decoder')
        assert vae.temporal_compress_level == sample_vae_config['temporal_compress_level']

    def test_vae_encode_shape(self, sample_vae_config, device, dtype):
        """Test VAE encoding produces correct output shape."""
        # Use smaller config for faster testing
        config = sample_vae_config.copy()
        config['encoder_config']['ch'] = 32
        config['encoder_config']['ch_mult'] = [1, 2]
        config['encoder_config']['num_res_blocks'] = 1
        config['decoder_config']['ch'] = 32
        config['decoder_config']['ch_mult'] = [1, 2]
        config['decoder_config']['num_res_blocks'] = 1

        vae = WanVAE_(**config).to(device).to(dtype)
        vae.eval()

        batch_size = 1
        channels = 3
        num_frames = 8
        height = 64
        width = 64

        x = torch.randn(batch_size, channels, num_frames, height, width, device=device, dtype=dtype)

        with torch.no_grad():
            encoded = vae.encode(x)

        # Check output shape after encoding
        z_channels = config['encoder_config']['z_channels']
        temporal_compress = config['temporal_compress_level']
        spatial_compress = 2 ** (len(config['encoder_config']['ch_mult']) - 1)

        expected_t = num_frames // temporal_compress
        expected_h = height // spatial_compress
        expected_w = width // spatial_compress

        assert encoded.shape == (batch_size, z_channels, expected_t, expected_h, expected_w)

    def test_vae_decode_shape(self, sample_vae_config, device, dtype):
        """Test VAE decoding produces correct output shape."""
        config = sample_vae_config.copy()
        config['encoder_config']['ch'] = 32
        config['encoder_config']['ch_mult'] = [1, 2]
        config['encoder_config']['num_res_blocks'] = 1
        config['decoder_config']['ch'] = 32
        config['decoder_config']['ch_mult'] = [1, 2]
        config['decoder_config']['num_res_blocks'] = 1

        vae = WanVAE_(**config).to(device).to(dtype)
        vae.eval()

        batch_size = 1
        z_channels = config['encoder_config']['z_channels']
        num_frames = 2
        height = 32
        width = 32

        z = torch.randn(batch_size, z_channels, num_frames, height, width, device=device, dtype=dtype)

        with torch.no_grad():
            decoded = vae.decode(z)

        # Check output shape after decoding
        out_channels = config['decoder_config']['out_ch']
        temporal_compress = config['temporal_compress_level']
        spatial_compress = 2 ** (len(config['decoder_config']['ch_mult']) - 1)

        expected_t = num_frames * temporal_compress
        expected_h = height * spatial_compress
        expected_w = width * spatial_compress

        assert decoded.shape == (batch_size, out_channels, expected_t, expected_h, expected_w)

    def test_vae_encode_decode_consistency(self, sample_vae_config, device, dtype):
        """Test that encode then decode produces similar output."""
        config = sample_vae_config.copy()
        config['encoder_config']['ch'] = 32
        config['encoder_config']['ch_mult'] = [1, 2]
        config['encoder_config']['num_res_blocks'] = 1
        config['decoder_config']['ch'] = 32
        config['decoder_config']['ch_mult'] = [1, 2]
        config['decoder_config']['num_res_blocks'] = 1

        vae = WanVAE_(**config).to(device).to(dtype)
        vae.eval()

        batch_size = 1
        channels = 3
        num_frames = 8
        height = 64
        width = 64

        x = torch.randn(batch_size, channels, num_frames, height, width, device=device, dtype=dtype)

        with torch.no_grad():
            encoded = vae.encode(x)
            decoded = vae.decode(encoded)

        # Decoded output should have same shape as input
        assert decoded.shape == x.shape

    def test_vae_no_nan_encode(self, sample_vae_config, device, dtype):
        """Test that VAE encoding doesn't produce NaN values."""
        config = sample_vae_config.copy()
        config['encoder_config']['ch'] = 32
        config['encoder_config']['ch_mult'] = [1, 2]
        config['encoder_config']['num_res_blocks'] = 1
        config['decoder_config']['ch'] = 32
        config['decoder_config']['ch_mult'] = [1, 2]
        config['decoder_config']['num_res_blocks'] = 1

        vae = WanVAE_(**config).to(device).to(dtype)
        vae.eval()

        batch_size = 1
        channels = 3
        num_frames = 8
        height = 64
        width = 64

        x = torch.randn(batch_size, channels, num_frames, height, width, device=device, dtype=dtype)

        with torch.no_grad():
            encoded = vae.encode(x)

        assert not torch.isnan(encoded).any()
        assert not torch.isinf(encoded).any()

    def test_vae_no_nan_decode(self, sample_vae_config, device, dtype):
        """Test that VAE decoding doesn't produce NaN values."""
        config = sample_vae_config.copy()
        config['encoder_config']['ch'] = 32
        config['encoder_config']['ch_mult'] = [1, 2]
        config['encoder_config']['num_res_blocks'] = 1
        config['decoder_config']['ch'] = 32
        config['decoder_config']['ch_mult'] = [1, 2]
        config['decoder_config']['num_res_blocks'] = 1

        vae = WanVAE_(**config).to(device).to(dtype)
        vae.eval()

        batch_size = 1
        z_channels = config['encoder_config']['z_channels']
        num_frames = 2
        height = 32
        width = 32

        z = torch.randn(batch_size, z_channels, num_frames, height, width, device=device, dtype=dtype)

        with torch.no_grad():
            decoded = vae.decode(z)

        assert not torch.isnan(decoded).any()
        assert not torch.isinf(decoded).any()

    @pytest.mark.parametrize("num_frames", [4, 8, 16])
    def test_vae_various_frame_counts(self, sample_vae_config, device, dtype, num_frames):
        """Test VAE with various frame counts."""
        config = sample_vae_config.copy()
        config['encoder_config']['ch'] = 32
        config['encoder_config']['ch_mult'] = [1, 2]
        config['encoder_config']['num_res_blocks'] = 1
        config['decoder_config']['ch'] = 32
        config['decoder_config']['ch_mult'] = [1, 2]
        config['decoder_config']['num_res_blocks'] = 1

        vae = WanVAE_(**config).to(device).to(dtype)
        vae.eval()

        batch_size = 1
        channels = 3
        height = 64
        width = 64

        x = torch.randn(batch_size, channels, num_frames, height, width, device=device, dtype=dtype)

        with torch.no_grad():
            encoded = vae.encode(x)
            decoded = vae.decode(encoded)

        assert decoded.shape == x.shape
        assert not torch.isnan(decoded).any()

    def test_vae_eval_mode(self, sample_vae_config):
        """Test that VAE can be set to eval mode."""
        with torch.device('meta'):
            vae = WanVAE_(**sample_vae_config)

        vae.eval()
        assert not vae.training

    def test_vae_config_attributes(self, sample_vae_config):
        """Test that VAE has correct configuration attributes."""
        with torch.device('meta'):
            vae = WanVAE_(**sample_vae_config)

        assert hasattr(vae, 'temporal_compress_level')
        assert vae.temporal_compress_level == sample_vae_config['temporal_compress_level']
