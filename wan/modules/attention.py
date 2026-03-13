# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import math

try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]

DEBUG_ATTENTION = True


def log_debug(message):
    if DEBUG_ATTENTION:
        print(f"[DEBUG] {message}")


def manual_attention(
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
):
    """Attention manuelle optimisée pour tous les devices"""
    # Déplacement immédiat sur le bon device
    device = q.device
    k = k.to(device)
    v = v.to(device)
    if q_lens is not None: q_lens = q_lens.to(device)
    if k_lens is not None: k_lens = k_lens.to(device)

    B, Lq, N, C = q.shape
    _, Lk, _, _ = k.shape
    original_dtype = q.dtype

    # Conversion au dtype de calcul
    q = q.to(dtype).transpose(1, 2)
    k = k.to(dtype).transpose(1, 2)
    v = v.to(dtype).transpose(1, 2)

    # Scaling
    scale_factor = softmax_scale or (1.0 / math.sqrt(C))
    if q_scale is not None:
        q = q * q_scale.view(1, -1, 1, 1)

    # Calcul des scores d'attention
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor

    # Création des masques
    attn_mask = torch.zeros(B, 1, Lq, Lk, device=device, dtype=torch.float32)

    # Masque de padding des clés
    if k_lens is not None:
        key_mask = torch.arange(Lk, device=device)[None, :] < k_lens[:, None]
        attn_mask = attn_mask.masked_fill(~key_mask.view(B, 1, 1, Lk), float('-inf'))

    # Masque causal
    if causal:
        causal_mask = torch.ones(Lq, Lk, device=device, dtype=torch.bool).tril()
        attn_mask = attn_mask.masked_fill(~causal_mask, float('-inf'))

    # Masque de fenêtre
    if window_size != (-1, -1):
        left, right = window_size
        row = torch.arange(Lq, device=device)[:, None]
        col = torch.arange(Lk, device=device)[None, :]
        window_mask = (row - col >= -left) & (row - col <= right)
        attn_mask = attn_mask.masked_fill(~window_mask, float('-inf'))

    # Application du masque
    attn_scores += attn_mask

    # Softmax et dropout
    attn_weights = torch.softmax(attn_scores, dim=-1)
    if not deterministic and dropout_p > 0:
        attn_weights = torch.dropout(attn_weights, dropout_p, True)

    # Calcul de la sortie
    out = torch.matmul(attn_weights, v)

    # Masque de padding des requêtes
    if q_lens is not None:
        query_mask = torch.arange(Lq, device=device)[None, :] < q_lens[:, None]
        out = out * query_mask.view(B, 1, Lq, 1).to(out.dtype)

    # Retour au format original
    return out.transpose(1, 2).contiguous().to(original_dtype)


def flash_attention(
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        version=None,
):
    """Wrapper pour FlashAttention avec fallback manuel"""
    # Fallback si FlashAttention non disponible
    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        return manual_attention(
            q, k, v, q_lens, k_lens, dropout_p, softmax_scale,
            q_scale, causal, window_size, deterministic, dtype
        )

    # Paramètres GPU
    device = q.device
    b, lq, lk = q.size(0), q.size(1), k.size(1)
    out_dtype = q.dtype

    # Préparation des séquences
    if q_lens is None:
        q_lens = torch.full((b,), lq, dtype=torch.int32, device=device)
        q_flat = q.flatten(0, 1)
    else:
        q_lens = q_lens.to(device)
        q_flat = torch.cat([u[:l] for u, l in zip(q, q_lens)])

    if k_lens is None:
        k_lens = torch.full((b,), lk, dtype=torch.int32, device=device)
        k_flat = k.flatten(0, 1)
        v_flat = v.flatten(0, 1)
    else:
        k_lens = k_lens.to(device)
        k_flat = torch.cat([u[:l] for u, l in zip(k, k_lens)])
        v_flat = torch.cat([u[:l] for u, l in zip(v, k_lens)])

    # Conversion de type
    q_flat = q_flat.to(dtype)
    k_flat = k_flat.to(dtype)
    v_flat = v_flat.to(dtype)

    # Application de q_scale
    if q_scale is not None:
        q_flat = q_flat * q_scale

    # Préparation des séquences cumulatives
    cu_seqlens_q = torch.cat([torch.tensor([0], device=device), q_lens.cumsum(0)])
    cu_seqlens_k = torch.cat([torch.tensor([0], device=device), k_lens.cumsum(0)])

    # Appel à FlashAttention
    try:
        if FLASH_ATTN_3_AVAILABLE and (version is None or version == 3):
            x = flash_attn_interface.flash_attn_varlen_func(
                q_flat, k_flat, v_flat,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q=lq, max_seqlen_k=lk,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic
            )[0]
        else:
            x = flash_attn.flash_attn_varlen_func(
                q_flat, k_flat, v_flat,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q=lq, max_seqlen_k=lk,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic
            )
        return x.unflatten(0, (b, lq)).to(out_dtype)
    except Exception as e:
        warnings.warn(f"FlashAttention failed: {e}, using manual attention")
        return manual_attention(
            q, k, v, q_lens, k_lens, dropout_p, softmax_scale,
            q_scale, causal, window_size, deterministic, dtype
        )


def attention(
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        fa_version=None,
):
    """Fonction d'attention unifiée"""
    # Synchronisation des devices
    device = q.device
    k = k.to(device)
    v = v.to(device)
    if q_lens is not None: q_lens = q_lens.to(device)
    if k_lens is not None: k_lens = k_lens.to(device)

    # Sélection de l'implémentation
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q, k, v, q_lens, k_lens, dropout_p, softmax_scale,
            q_scale, causal, window_size, deterministic, dtype, fa_version
        )
    else:
        return manual_attention(
            q, k, v, q_lens, k_lens, dropout_p, softmax_scale,
            q_scale, causal, window_size, deterministic, dtype
        )