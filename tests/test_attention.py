"""
Unit tests for attention mechanisms in Wan2.1.

Copyright (c) 2025 Kuaishou. All rights reserved.
"""

import pytest
import torch
from wan.modules.attention import attention


class TestAttention:
    """Test suite for attention mechanisms."""

    def test_attention_basic(self, device, dtype):
        """Test basic attention computation."""
        batch_size = 2
        seq_len = 16
        num_heads = 4
        head_dim = 64

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

        output = attention(q, k, v)

        assert output.shape == (batch_size, seq_len, num_heads, head_dim)
        assert output.dtype == dtype
        assert output.device == device
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_attention_with_mask(self, device, dtype):
        """Test attention with causal mask."""
        batch_size = 2
        seq_len = 16
        num_heads = 4
        head_dim = 64

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

        output = attention(q, k, v, mask=mask)

        assert output.shape == (batch_size, seq_len, num_heads, head_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_attention_different_seq_lengths(self, device, dtype):
        """Test attention with different query and key/value sequence lengths."""
        batch_size = 2
        q_seq_len = 8
        kv_seq_len = 16
        num_heads = 4
        head_dim = 64

        q = torch.randn(batch_size, q_seq_len, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, kv_seq_len, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, kv_seq_len, num_heads, head_dim, device=device, dtype=dtype)

        output = attention(q, k, v)

        assert output.shape == (batch_size, q_seq_len, num_heads, head_dim)
        assert not torch.isnan(output).any()

    def test_attention_zero_values(self, device, dtype):
        """Test attention with zero inputs."""
        batch_size = 1
        seq_len = 8
        num_heads = 2
        head_dim = 32

        q = torch.zeros(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.zeros(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.zeros(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

        output = attention(q, k, v)

        assert output.shape == (batch_size, seq_len, num_heads, head_dim)
        # With zero inputs, output should be zero or close to zero
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-5)

    def test_attention_batch_size_one(self, device, dtype):
        """Test attention with batch size of 1."""
        batch_size = 1
        seq_len = 32
        num_heads = 8
        head_dim = 64

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

        output = attention(q, k, v)

        assert output.shape == (batch_size, seq_len, num_heads, head_dim)
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("seq_len", [1, 8, 32, 128])
    def test_attention_various_seq_lengths(self, device, dtype, seq_len):
        """Test attention with various sequence lengths."""
        batch_size = 2
        num_heads = 4
        head_dim = 64

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

        output = attention(q, k, v)

        assert output.shape == (batch_size, seq_len, num_heads, head_dim)
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("num_heads", [1, 2, 4, 8, 16])
    def test_attention_various_num_heads(self, device, dtype, num_heads):
        """Test attention with various numbers of heads."""
        batch_size = 2
        seq_len = 16
        head_dim = 64

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

        output = attention(q, k, v)

        assert output.shape == (batch_size, seq_len, num_heads, head_dim)
        assert not torch.isnan(output).any()

    def test_attention_gradient_flow(self, device, dtype):
        """Test that gradients flow properly through attention."""
        if dtype == torch.bfloat16:
            pytest.skip("Gradient checking not supported for bfloat16")

        batch_size = 2
        seq_len = 8
        num_heads = 2
        head_dim = 32

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)

        output = attention(q, k, v)
        loss = output.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert not torch.isnan(q.grad).any()
        assert not torch.isnan(k.grad).any()
        assert not torch.isnan(v.grad).any()
