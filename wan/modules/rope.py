from typing import Optional, Tuple

import triton
import triton.language as tl
import torch


def pad_tensor(
    original_tensor: torch.tensor, target_len: int, pad_value: float = 0.0
) -> torch.tensor:
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.full(
        (pad_size, s1, s2),
        pad_value,
        dtype=original_tensor.dtype,
        device=original_tensor.device,
    )
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


def rope_apply_pytorch(
    x: torch.tensor,
    grid_sizes: torch.tensor,
    freqs: Tuple[torch.tensor],
    sp_size: Optional[int] = None,
    sp_rank: Optional[int] = None,
) -> torch.tensor:
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    c0 = c - 2 * (c // 3)
    c1 = c // 3
    c2 = c // 3

    # split freqs
    freqs_real = freqs[0].split([c0, c1, c2], dim=1)
    freqs_imag = freqs[-1].split([c0, c1, c2], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = x[i, :seq_len].reshape(s, n, -1, 2)
        x_real = x_i[..., 0]
        x_imag = x_i[..., 1]
        freqs_real = torch.cat(
            [
                freqs_real[0][:f].view(f, 1, 1, c0).expand(f, h, w, c0),
                freqs_real[1][:h].view(1, h, 1, c1).expand(f, h, w, c1),
                freqs_real[2][:w].view(1, 1, w, c2).expand(f, h, w, c2),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        freqs_imag = torch.cat(
            [
                freqs_imag[0][:f].view(f, 1, 1, c0).expand(f, h, w, c0),
                freqs_imag[1][:h].view(1, h, 1, c1).expand(f, h, w, c1),
                freqs_imag[2][:w].view(1, 1, w, c2).expand(f, h, w, c2),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        if sp_rank is None:
            freqs_real_rank = freqs_real
            freqs_imag_rank = freqs_imag
        else:
            freqs_real = pad_tensor(freqs_real, s * sp_size, 1.0)
            freqs_imag = pad_tensor(freqs_imag, s * sp_size, 0.0)
            freqs_real_rank = freqs_real[(sp_rank * s) : ((sp_rank + 1) * s), :, :]
            freqs_imag_rank = freqs_imag[(sp_rank * s) : ((sp_rank + 1) * s), :, :]

        out_real = x_real * freqs_real_rank - x_imag * freqs_imag_rank
        out_imag = x_real * freqs_imag_rank + x_imag * freqs_real_rank

        x_out = torch.stack([out_real, out_imag], dim=-1).flatten(2)
        x_out = torch.cat([x_out, x[i, seq_len:]], dim=0)

        # append to collection
        output.append(x_out)

    return torch.stack(output)


@triton.jit
def rope_kernel(
    x_ptr,  # [B, S, N, 2C]
    grid_sizes_ptr,  # [B, 3]
    freqs_real_ptr,  # [M, C]
    freqs_imag_ptr,  # [M, C]
    output_ptr,  # [B, S, N, 2C]
    sp_size,  # SP world size
    sp_rank,  # SP rank
    B,
    S,
    N: tl.constexpr,
    C: tl.constexpr,
    M: tl.constexpr,
    CfM: tl.constexpr,
    ChM: tl.constexpr,
    CwM: tl.constexpr,
    SEQ_BLOCK: tl.constexpr,
    HEADS_BLOCK: tl.constexpr,
):
    Cf = C - 2 * (C // 3)
    Ch = C // 3
    Cw = C // 3

    batch_idx = tl.program_id(0)
    seqlen_group_idx = tl.program_id(1)
    head_group_idx = tl.program_id(2)

    base = batch_idx * 3
    F = tl.load(grid_sizes_ptr + base + 0)
    H = tl.load(grid_sizes_ptr + base + 1)
    W = tl.load(grid_sizes_ptr + base + 2)
    seq_len = F * H * W

    global_offset = sp_rank * S + seqlen_group_idx * SEQ_BLOCK
    seq_indices = global_offset + tl.arange(0, SEQ_BLOCK)

    limit = tl.minimum(seq_len, S * sp_size)
    seq_mask = seq_indices < limit
    seq_indices = tl.where(seq_mask, seq_indices, 0)

    HW = H * W
    f_idx = seq_indices // HW
    rem = seq_indices - f_idx * HW
    h_idx = rem // W
    w_idx = rem - h_idx * W

    freq_offset_cf = tl.arange(0, CfM)  # 第1段列偏移 [0, Cf)
    freq_offset_ch = Cf + tl.arange(0, ChM)  # 第2段列偏移 [Cf, Cf+Ch)
    freq_offset_cw = Cf + Ch + tl.arange(0, CwM)  # 第3段列偏移 [Cf+Ch, C)
    # 按照每个序列位置取对应频率值 (利用广播计算每个位置不同行的值)
    # 频率表取值地址 = idx * C + col_offset
    freq_addr_cf = f_idx[:, None] * C + freq_offset_cf[None, :]
    freq_addr_ch = h_idx[:, None] * C + freq_offset_ch[None, :]
    freq_addr_cw = w_idx[:, None] * C + freq_offset_cw[None, :]

    freqs_real_cf = tl.load(
        freqs_real_ptr + freq_addr_cf,
        mask=(seq_mask[:, None] & (freq_offset_cf[None, :] < Cf)),
        other=1.0,
    ).to(tl.float32)
    freqs_imag_cf = tl.load(
        freqs_imag_ptr + freq_addr_cf,
        mask=(seq_mask[:, None] & (freq_offset_cf[None, :] < Cf)),
        other=1.0,
    ).to(tl.float32)
    freqs_real_ch = tl.load(
        freqs_real_ptr + freq_addr_ch,
        mask=(seq_mask[:, None] & (freq_offset_ch[None, :] < Cf + Ch)),
        other=1.0,
    ).to(tl.float32)
    freqs_imag_ch = tl.load(
        freqs_imag_ptr + freq_addr_ch,
        mask=(seq_mask[:, None] & (freq_offset_ch[None, :] < Cf + Ch)),
        other=1.0,
    ).to(tl.float32)
    freqs_real_cw = tl.load(
        freqs_real_ptr + freq_addr_cw,
        mask=(seq_mask[:, None] & (freq_offset_cw[None, :] < C)),
        other=1.0,
    ).to(tl.float32)
    freqs_imag_cw = tl.load(
        freqs_imag_ptr + freq_addr_cw,
        mask=(seq_mask[:, None] & (freq_offset_cw[None, :] < C)),
        other=1.0,
    ).to(tl.float32)

    # 将频率值扩展维度以便与x相乘 (在head维度上广播)
    freqs_real_cf = freqs_real_cf[:, None, :]  # [SEQ_BLOCK, 1, Cf]
    freqs_imag_cf = freqs_imag_cf[:, None, :]
    freqs_real_ch = freqs_real_ch[:, None, :]
    freqs_imag_ch = freqs_imag_ch[:, None, :]
    freqs_real_cw = freqs_real_cw[:, None, :]
    freqs_imag_cw = freqs_imag_cw[:, None, :]

    # 加载输入x对应块的实部和虚部 (形状: [SEQ_BLOCK, HEADS_BLOCK, C])
    seq_offset = seqlen_group_idx * SEQ_BLOCK + tl.arange(0, SEQ_BLOCK)
    head_offset = head_group_idx * HEADS_BLOCK + tl.arange(0, HEADS_BLOCK)
    # 计算x_ptr偏移地址
    base_offset = batch_idx * S * N * 2 * C
    seq_head_offset = (
        base_offset
        + seq_offset[:, None, None] * (N * 2 * C)
        + head_offset[None, :, None] * (2 * C)
    )
    x_mask = (seq_offset < S)[:, None, None] & (head_offset < N)[None, :, None]

    # 加载输入 x 的对应通道段数据，超出实际长度部分掩码为0
    # 段1：通道 [0, Cf-1]
    chan_cf = tl.arange(0, CfM * 2)
    mask_2cf_chan = chan_cf < Cf * 2
    x_cf = tl.load(
        x_ptr + seq_head_offset + chan_cf[None, None, :],
        mask=(x_mask & mask_2cf_chan[None, None, :]),
        other=0.0,
    ).to(tl.float32)
    x_cf = x_cf.reshape(
        SEQ_BLOCK, HEADS_BLOCK, CfM, 2
    )  # [SEQ_BLOCK, HEADS_BLOCK, CfM, 2]
    x_real_cf, x_imag_cf = x_cf.split()

    # 计算 RoPE 旋转（段1）
    out_real_cf = x_real_cf * freqs_real_cf - x_imag_cf * freqs_imag_cf
    out_imag_cf = x_real_cf * freqs_imag_cf + x_imag_cf * freqs_real_cf

    out_cf = tl.interleave(out_real_cf, out_imag_cf)  # [SEQ_BLOCK, HEADS_BLOCK, CfM, 2]
    tl.store(
        output_ptr + seq_head_offset + chan_cf[None, None, :],
        out_cf,
        mask=(x_mask & mask_2cf_chan[None, None, :]),
    )

    # 段2：通道 [Cf, Cf+Ch-1]
    chan_ch = tl.arange(0, ChM * 2) + Cf * 2
    mask_2ch_chan = chan_ch < 2 * (Cf + Ch)
    x_ch = tl.load(
        x_ptr + seq_head_offset + chan_ch[None, None, :],
        mask=(x_mask & mask_2ch_chan[None, None, :]),
        other=0.0,
    ).to(tl.float32)
    x_ch = x_ch.reshape(SEQ_BLOCK, HEADS_BLOCK, ChM, 2)
    x_real_ch, x_imag_ch = x_ch.split()
    out_real_ch = x_real_ch * freqs_real_ch - x_imag_ch * freqs_imag_ch
    out_imag_ch = x_real_ch * freqs_imag_ch + x_imag_ch * freqs_real_ch

    out_ch = tl.interleave(out_real_ch, out_imag_ch)  # [SEQ_BLOCK, HEADS_BLOCK, ChM, 2]
    tl.store(
        output_ptr + seq_head_offset + chan_ch[None, None, :],
        out_ch,
        mask=(x_mask & mask_2ch_chan[None, None, :]),
    )

    # 段3：通道 [Cf+Ch, C-1]
    chan_cw = tl.arange(0, CwM * 2) + (Cf + Ch) * 2
    mask_2cw_chan = chan_cw < 2 * C
    x_cw = tl.load(
        x_ptr + seq_head_offset + chan_cw[None, None, :],
        mask=(x_mask & mask_2cw_chan[None, None, :]),
        other=0.0,
    ).to(tl.float32)
    x_cw = x_cw.reshape(SEQ_BLOCK, HEADS_BLOCK, CwM, 2)
    x_real_cw, x_imag_cw = x_cw.split()
    out_real_cw = x_real_cw * freqs_real_cw - x_imag_cw * freqs_imag_cw
    out_imag_cw = x_real_cw * freqs_imag_cw + x_imag_cw * freqs_real_cw

    out_cw = tl.interleave(out_real_cw, out_imag_cw)
    tl.store(
        output_ptr + seq_head_offset + chan_cw[None, None, :],
        out_cw,
        mask=(x_mask & mask_2cw_chan[None, None, :]),
    )


@torch._dynamo.disable
def rope_apply_triton(
    x: torch.tensor,
    grid_sizes: torch.tensor,
    freqs: Tuple[torch.tensor],
    sp_size: Optional[int] = None,
    sp_rank: Optional[int] = None,
) -> torch.tensor:
    """
    x: [1, 9450, 40, 128]
    grid_sizes: [[21, 45, 80]]
    freqs_real: [1024, 64]
    freqs_imag: [1024, 64]
    """
    B, S, N, C = x.shape
    C = C // 2
    Cf = C - 2 * (C // 3)  # 第一维度频率长度
    Ch = C // 3  # 第二维度频率长度
    Cw = C // 3  # 第三维度频率长度
    M = freqs[0].shape[0]

    SEQ_BLOCK = 64  # 每个线程块处理的序列长度
    HEADS_BLOCK = 8  # 每个线程块处理的头数

    if sp_rank is None:
        sp_size = 1
        sp_rank = 0

    grid_sizes = grid_sizes.to(device=x.device)
    output = torch.empty_like(x)

    rope_kernel[(B, triton.cdiv(S, SEQ_BLOCK), triton.cdiv(N, HEADS_BLOCK))](
        x,
        grid_sizes,
        freqs[0],
        freqs[-1],
        output,
        sp_size,
        sp_rank,
        B,
        S,
        N=N,
        C=C,
        M=M,
        CfM=triton.next_power_of_2(Cf),
        ChM=triton.next_power_of_2(Ch),
        CwM=triton.next_power_of_2(Cw),
        SEQ_BLOCK=SEQ_BLOCK,
        HEADS_BLOCK=HEADS_BLOCK,
        num_warps=32,
        num_stages=3,
    )

    return output.float()
