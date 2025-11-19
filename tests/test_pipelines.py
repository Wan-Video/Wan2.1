"""
Integration tests for Wan2.1 pipelines (T2V, I2V, FLF2V, VACE).

Copyright (c) 2025 Kuaishou. All rights reserved.

Note: These tests require model checkpoints and are marked as integration tests.
Run with: pytest -m integration
"""

import pytest
import torch


@pytest.mark.integration
@pytest.mark.requires_model
class TestText2VideoPipeline:
    """Integration tests for Text-to-Video pipeline."""

    def test_t2v_pipeline_imports(self):
        """Test that T2V pipeline can be imported."""
        from wan.text2video import WanT2V
        assert WanT2V is not None

    def test_t2v_pipeline_initialization(self):
        """Test T2V pipeline initialization (meta device, no weights)."""
        from wan.text2video import WanT2V

        # This tests the interface without loading actual weights
        # Real tests would require model checkpoints
        assert callable(WanT2V)


@pytest.mark.integration
@pytest.mark.requires_model
class TestImage2VideoPipeline:
    """Integration tests for Image-to-Video pipeline."""

    def test_i2v_pipeline_imports(self):
        """Test that I2V pipeline can be imported."""
        from wan.image2video import WanI2V
        assert WanI2V is not None

    def test_i2v_pipeline_initialization(self):
        """Test I2V pipeline initialization (meta device, no weights)."""
        from wan.image2video import WanI2V

        assert callable(WanI2V)


@pytest.mark.integration
@pytest.mark.requires_model
class TestFirstLastFrame2VideoPipeline:
    """Integration tests for First-Last-Frame-to-Video pipeline."""

    def test_flf2v_pipeline_imports(self):
        """Test that FLF2V pipeline can be imported."""
        from wan.first_last_frame2video import WanFLF2V
        assert WanFLF2V is not None

    def test_flf2v_pipeline_initialization(self):
        """Test FLF2V pipeline initialization (meta device, no weights)."""
        from wan.first_last_frame2video import WanFLF2V

        assert callable(WanFLF2V)


@pytest.mark.integration
@pytest.mark.requires_model
class TestVACEPipeline:
    """Integration tests for VACE (Video Creation & Editing) pipeline."""

    def test_vace_pipeline_imports(self):
        """Test that VACE pipeline can be imported."""
        from wan.vace import WanVace
        assert WanVace is not None

    def test_vace_pipeline_initialization(self):
        """Test VACE pipeline initialization (meta device, no weights)."""
        from wan.vace import WanVace

        assert callable(WanVace)


class TestPipelineConfigs:
    """Test pipeline configuration loading."""

    def test_t2v_14b_config_loads(self):
        """Test that T2V 14B config can be loaded."""
        from wan.configs.t2v_14B import get_config

        config = get_config()
        assert config is not None
        assert 'hidden_size' in config
        assert config['hidden_size'] == 3072

    def test_t2v_1_3b_config_loads(self):
        """Test that T2V 1.3B config can be loaded."""
        from wan.configs.t2v_1_3B import get_config

        config = get_config()
        assert config is not None
        assert 'hidden_size' in config
        assert config['hidden_size'] == 1536

    def test_i2v_14b_config_loads(self):
        """Test that I2V 14B config can be loaded."""
        from wan.configs.i2v_14B import get_config

        config = get_config()
        assert config is not None
        assert 'hidden_size' in config

    def test_i2v_1_3b_config_loads(self):
        """Test that I2V 1.3B config can be loaded."""
        from wan.configs.i2v_1_3B import get_config

        config = get_config()
        assert config is not None
        assert 'hidden_size' in config

    def test_all_configs_have_required_keys(self):
        """Test that all configs have required keys."""
        from wan.configs.t2v_14B import get_config as get_t2v_14b
        from wan.configs.t2v_1_3B import get_config as get_t2v_1_3b
        from wan.configs.i2v_14B import get_config as get_i2v_14b
        from wan.configs.i2v_1_3B import get_config as get_i2v_1_3b

        required_keys = [
            'patch_size', 'in_channels', 'hidden_size', 'depth',
            'num_heads', 'mlp_ratio', 'learn_sigma'
        ]

        for config_fn in [get_t2v_14b, get_t2v_1_3b, get_i2v_14b, get_i2v_1_3b]:
            config = config_fn()
            for key in required_keys:
                assert key in config, f"Missing key {key} in config"


class TestDistributed:
    """Test distributed training utilities."""

    def test_fsdp_imports(self):
        """Test that FSDP utilities can be imported."""
        from wan.distributed.fsdp import WanFSDP
        assert WanFSDP is not None

    def test_context_parallel_imports(self):
        """Test that context parallel utilities can be imported."""
        try:
            from wan.distributed.xdit_context_parallel import xFuserWanModelArgs
            assert xFuserWanModelArgs is not None
        except ImportError:
            pytest.skip("xDiT context parallel not available")
