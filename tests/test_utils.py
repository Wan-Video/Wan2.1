"""
Unit tests for utility functions in Wan2.1.

Copyright (c) 2025 Kuaishou. All rights reserved.
"""

import pytest
import torch
import tempfile
from pathlib import Path
from wan.utils.utils import video_to_torch_cached, image_to_torch_cached


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_image_to_torch_cached_basic(self, temp_dir):
        """Test basic image loading and caching."""
        # Create a dummy image file using PIL
        try:
            from PIL import Image
            import numpy as np

            # Create a simple test image
            img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = temp_dir / "test_image.png"
            img.save(img_path)

            # Load image with caching
            tensor = image_to_torch_cached(str(img_path))

            assert isinstance(tensor, torch.Tensor)
            assert tensor.ndim == 3  # CHW format
            assert tensor.shape[0] == 3  # RGB channels
            assert tensor.dtype == torch.float32
            assert tensor.min() >= 0.0
            assert tensor.max() <= 1.0

        except ImportError:
            pytest.skip("PIL not available")

    def test_image_to_torch_cached_resize(self, temp_dir):
        """Test image loading with resizing."""
        try:
            from PIL import Image
            import numpy as np

            # Create a test image
            img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = temp_dir / "test_image.png"
            img.save(img_path)

            # Load and resize
            target_size = (64, 64)
            tensor = image_to_torch_cached(str(img_path), size=target_size)

            assert tensor.shape[1] == target_size[0]  # height
            assert tensor.shape[2] == target_size[1]  # width

        except ImportError:
            pytest.skip("PIL not available")

    def test_image_to_torch_nonexistent_file(self):
        """Test that loading a nonexistent image raises an error."""
        with pytest.raises(Exception):
            image_to_torch_cached("/nonexistent/path/image.png")

    def test_video_to_torch_cached_basic(self, temp_dir):
        """Test basic video loading (if av is available)."""
        try:
            import av
            import numpy as np

            # Create a simple test video
            video_path = temp_dir / "test_video.mp4"
            container = av.open(str(video_path), mode='w')
            stream = container.add_stream('mpeg4', rate=30)
            stream.width = 64
            stream.height = 64
            stream.pix_fmt = 'yuv420p'

            for i in range(10):
                frame = av.VideoFrame(64, 64, 'rgb24')
                frame_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                frame.planes[0].update(frame_array)
                packet = stream.encode(frame)
                if packet:
                    container.mux(packet)

            # Flush remaining packets
            for packet in stream.encode():
                container.mux(packet)

            container.close()

            # Load video with caching
            tensor = video_to_torch_cached(str(video_path))

            assert isinstance(tensor, torch.Tensor)
            assert tensor.ndim == 4  # TCHW format
            assert tensor.shape[1] == 3  # RGB channels
            assert tensor.dtype == torch.float32

        except ImportError:
            pytest.skip("av library not available")

    def test_video_to_torch_nonexistent_file(self):
        """Test that loading a nonexistent video raises an error."""
        with pytest.raises(Exception):
            video_to_torch_cached("/nonexistent/path/video.mp4")


class TestPromptExtension:
    """Test suite for prompt extension utilities."""

    def test_prompt_extend_imports(self):
        """Test that prompt extension modules can be imported."""
        try:
            from wan.utils.prompt_extend import extend_prompt_with_qwen, extend_prompt_with_dashscope
            assert extend_prompt_with_qwen is not None
            assert extend_prompt_with_dashscope is not None
        except ImportError as e:
            pytest.fail(f"Failed to import prompt extension: {e}")

    def test_prompt_extend_qwen_basic(self):
        """Test basic Qwen prompt extension (without model)."""
        try:
            from wan.utils.prompt_extend import extend_prompt_with_qwen

            # This will likely fail without a model, but we're testing the interface
            # In a real test, you'd mock the model
            assert callable(extend_prompt_with_qwen)

        except ImportError:
            pytest.skip("Prompt extension not available")


class TestFMSolvers:
    """Test suite for flow matching solvers."""

    def test_fm_solver_imports(self):
        """Test that FM solver modules can be imported."""
        from wan.utils.fm_solvers import FlowMatchingDPMSolver
        from wan.utils.fm_solvers_unipc import FlowMatchingUniPCSolver

        assert FlowMatchingDPMSolver is not None
        assert FlowMatchingUniPCSolver is not None

    def test_dpm_solver_initialization(self):
        """Test DPM solver initialization."""
        from wan.utils.fm_solvers import FlowMatchingDPMSolver

        solver = FlowMatchingDPMSolver(
            num_steps=20,
            order=2,
            skip_type='time_uniform',
            method='multistep',
        )

        assert solver is not None
        assert solver.num_steps == 20

    def test_unipc_solver_initialization(self):
        """Test UniPC solver initialization."""
        from wan.utils.fm_solvers_unipc import FlowMatchingUniPCSolver

        solver = FlowMatchingUniPCSolver(
            num_steps=20,
            order=2,
            skip_type='time_uniform',
        )

        assert solver is not None
        assert solver.num_steps == 20

    def test_solver_get_timesteps(self):
        """Test that solver can generate timesteps."""
        from wan.utils.fm_solvers import FlowMatchingDPMSolver

        solver = FlowMatchingDPMSolver(
            num_steps=10,
            order=2,
        )

        timesteps = solver.get_time_steps()

        assert len(timesteps) > 0
        assert all(0 <= t <= 1 for t in timesteps)
