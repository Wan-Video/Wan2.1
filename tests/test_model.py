"""
Unit tests for WanModel (DiT) in Wan2.1.

Copyright (c) 2025 Kuaishou. All rights reserved.
"""

import pytest
import torch
from wan.modules.model import WanModel


class TestWanModel:
    """Test suite for WanModel (Diffusion Transformer)."""

    def test_model_initialization_1_3b(self, sample_config_1_3b, device):
        """Test 1.3B model initialization."""
        with torch.device('meta'):
            model = WanModel(**sample_config_1_3b)

        assert model is not None
        assert model.hidden_size == 1536
        assert model.depth == 20
        assert model.num_heads == 24

    def test_model_initialization_14b(self, sample_config_14b, device):
        """Test 14B model initialization."""
        with torch.device('meta'):
            model = WanModel(**sample_config_14b)

        assert model is not None
        assert model.hidden_size == 3072
        assert model.depth == 42
        assert model.num_heads == 24

    def test_model_forward_shape_small(self, sample_config_1_3b, device, dtype):
        """Test forward pass with small model on small input (CPU compatible)."""
        # Use smaller config for faster testing
        config = sample_config_1_3b.copy()
        config['hidden_size'] = 256
        config['depth'] = 2
        config['num_heads'] = 4

        model = WanModel(**config).to(device).to(dtype)
        model.eval()

        batch_size = 1
        num_frames = 4
        height = 16
        width = 16
        in_channels = config['in_channels']
        text_len = config['text_len']
        t5_feat_dim = config['t5_feat_dim']
        cap_feat_dim = config['cap_feat_dim']

        # Create dummy inputs
        x = torch.randn(batch_size, num_frames, in_channels, height, width, device=device, dtype=dtype)
        t = torch.randn(batch_size, device=device, dtype=dtype)
        y = torch.randn(batch_size, 1, cap_feat_dim, device=device, dtype=dtype)
        mask = torch.ones(batch_size, text_len, device=device, dtype=torch.bool)
        txt_fea = torch.randn(batch_size, text_len, t5_feat_dim, device=device, dtype=dtype)

        with torch.no_grad():
            output = model(x, t, y, mask, txt_fea)

        expected_shape = (batch_size, num_frames, in_channels, height, width)
        assert output.shape == expected_shape
        assert output.dtype == dtype
        assert output.device == device

    def test_model_parameter_count_1_3b(self, sample_config_1_3b):
        """Test parameter count is reasonable for 1.3B model."""
        with torch.device('meta'):
            model = WanModel(**sample_config_1_3b)

        total_params = sum(p.numel() for p in model.parameters())
        # Should be around 1.3B parameters (allow some variance)
        assert 1.0e9 < total_params < 2.0e9, f"Expected ~1.3B params, got {total_params:,}"

    def test_model_parameter_count_14b(self, sample_config_14b):
        """Test parameter count is reasonable for 14B model."""
        with torch.device('meta'):
            model = WanModel(**sample_config_14b)

        total_params = sum(p.numel() for p in model.parameters())
        # Should be around 14B parameters (allow some variance)
        assert 10e9 < total_params < 20e9, f"Expected ~14B params, got {total_params:,}"

    def test_model_no_nan_output(self, sample_config_1_3b, device, dtype):
        """Test that model output doesn't contain NaN values."""
        config = sample_config_1_3b.copy()
        config['hidden_size'] = 256
        config['depth'] = 2
        config['num_heads'] = 4

        model = WanModel(**config).to(device).to(dtype)
        model.eval()

        batch_size = 1
        num_frames = 4
        height = 16
        width = 16
        in_channels = config['in_channels']
        text_len = config['text_len']
        t5_feat_dim = config['t5_feat_dim']
        cap_feat_dim = config['cap_feat_dim']

        x = torch.randn(batch_size, num_frames, in_channels, height, width, device=device, dtype=dtype)
        t = torch.randn(batch_size, device=device, dtype=dtype)
        y = torch.randn(batch_size, 1, cap_feat_dim, device=device, dtype=dtype)
        mask = torch.ones(batch_size, text_len, device=device, dtype=torch.bool)
        txt_fea = torch.randn(batch_size, text_len, t5_feat_dim, device=device, dtype=dtype)

        with torch.no_grad():
            output = model(x, t, y, mask, txt_fea)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_model_eval_mode(self, sample_config_1_3b, device):
        """Test that model can be set to eval mode."""
        with torch.device('meta'):
            model = WanModel(**sample_config_1_3b)

        model.eval()
        assert not model.training

    def test_model_train_mode(self, sample_config_1_3b, device):
        """Test that model can be set to train mode."""
        with torch.device('meta'):
            model = WanModel(**sample_config_1_3b)

        model.train()
        assert model.training

    def test_model_config_attributes(self, sample_config_1_3b):
        """Test that model has correct configuration attributes."""
        with torch.device('meta'):
            model = WanModel(**sample_config_1_3b)

        assert hasattr(model, 'patch_size')
        assert hasattr(model, 'in_channels')
        assert hasattr(model, 'hidden_size')
        assert hasattr(model, 'depth')
        assert hasattr(model, 'num_heads')
        assert model.patch_size == sample_config_1_3b['patch_size']
        assert model.in_channels == sample_config_1_3b['in_channels']

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_model_various_batch_sizes(self, sample_config_1_3b, device, dtype, batch_size):
        """Test model with various batch sizes."""
        config = sample_config_1_3b.copy()
        config['hidden_size'] = 256
        config['depth'] = 2
        config['num_heads'] = 4

        model = WanModel(**config).to(device).to(dtype)
        model.eval()

        num_frames = 4
        height = 16
        width = 16
        in_channels = config['in_channels']
        text_len = config['text_len']
        t5_feat_dim = config['t5_feat_dim']
        cap_feat_dim = config['cap_feat_dim']

        x = torch.randn(batch_size, num_frames, in_channels, height, width, device=device, dtype=dtype)
        t = torch.randn(batch_size, device=device, dtype=dtype)
        y = torch.randn(batch_size, 1, cap_feat_dim, device=device, dtype=dtype)
        mask = torch.ones(batch_size, text_len, device=device, dtype=torch.bool)
        txt_fea = torch.randn(batch_size, text_len, t5_feat_dim, device=device, dtype=dtype)

        with torch.no_grad():
            output = model(x, t, y, mask, txt_fea)

        assert output.shape[0] == batch_size
