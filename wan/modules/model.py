# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import hashlib
import json
import math
import os
import shutil
import warnings
from pathlib import Path

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

try:
    from lite_linear import LiteLinear
except Exception as exc:  # pragma: no cover - optional dependency
    LiteLinear = None
    _LITELINEAR_IMPORT_ERROR = exc
    _LITELINEAR_PATCH_HELPERS_ERROR = exc
    _litelinear_default_cache_root = None
    _litelinear_resolve_or_build_patched_checkpoint = None
    _litelinear_resolve_strict_checkpoint_path = None
    _litelinear_safe_open_for_patch_build = None
else:
    _LITELINEAR_IMPORT_ERROR = None
    try:
        from lite_linear.checkpoint_patch_core import (
            default_cache_root as _litelinear_default_cache_root,
            resolve_or_build_patched_checkpoint as _litelinear_resolve_or_build_patched_checkpoint,
            resolve_strict_checkpoint_path as _litelinear_resolve_strict_checkpoint_path,
        )
        from lite_linear.patched_checkpoint import (
            get_safe_open_for_patch_build as _litelinear_safe_open_for_patch_build,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        _LITELINEAR_PATCH_HELPERS_ERROR = exc
        _litelinear_default_cache_root = None
        _litelinear_resolve_or_build_patched_checkpoint = None
        _litelinear_resolve_strict_checkpoint_path = None
        _litelinear_safe_open_for_patch_build = None
    else:
        _LITELINEAR_PATCH_HELPERS_ERROR = None

__all__ = ['WanModel']

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2
_BOOL_TRUE = {"1", "true", "yes", "y", "on"}


def _env_true(name, default="0"):
    return os.environ.get(name, default).strip().lower() in _BOOL_TRUE


def _litelinear_disabled():
    return _env_true("LITELINEAR_DISABLED", "0")


def _litelinear_online_patch_enabled():
    return (not _litelinear_disabled()) and _env_true("LITELINEAR_ONLINE_PATCH", "0")


def _litelinear_strict_mode_enabled():
    return (not _litelinear_disabled()) and (not _litelinear_online_patch_enabled())


def _litelinear_patch_tag():
    tag = os.environ.get("LITELINEAR_PATCH_TAG", "nocalib").strip().lower()
    return "calib" if tag == "calib" else "nocalib"


def _litelinear_patch_config_path():
    raw = os.environ.get("LITELINEAR_PATCH_CONFIG", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _sync_file_reference(src: Path, dst: Path):
    src = src.resolve()
    if dst.is_symlink():
        try:
            if dst.resolve() == src:
                return
        except FileNotFoundError:
            pass
        dst.unlink(missing_ok=True)
    elif dst.exists():
        if dst.resolve() == src:
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _litelinear_cached_patched_dir_ready(
        patched_dir: Path,
        *,
        index_name: str,
        config_name: str,
        original_weight_map: dict):
    patched_index_path = patched_dir / index_name
    patched_config_path = patched_dir / config_name
    if not patched_index_path.is_file() or not patched_config_path.exists():
        return False
    try:
        patched_data = json.loads(patched_index_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    patched_weight_map = patched_data.get("weight_map")
    if not isinstance(patched_weight_map, dict) or not patched_weight_map:
        return False
    # If the cached index is identical to the original one, no shard remap was
    # applied and the original checkpoint directory should still be used.
    if patched_weight_map == original_weight_map:
        return False
    for shard_name in {str(name) for name in patched_weight_map.values()}:
        if not (patched_dir / shard_name).is_file():
            return False
    return True


def _resolve_litelinear_pretrained_dir(pretrained_model_name_or_path):
    if LiteLinear is None or _LITELINEAR_PATCH_HELPERS_ERROR is not None:
        return pretrained_model_name_or_path
    if not (_litelinear_online_patch_enabled() or _litelinear_strict_mode_enabled()):
        return pretrained_model_name_or_path

    try:
        checkpoint_dir = Path(pretrained_model_name_or_path).expanduser().resolve()
    except Exception:
        return pretrained_model_name_or_path

    index_path = checkpoint_dir / "diffusion_pytorch_model.safetensors.index.json"
    config_path = checkpoint_dir / "config.json"
    if not checkpoint_dir.is_dir() or not index_path.is_file() or not config_path.is_file():
        return pretrained_model_name_or_path

    data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return pretrained_model_name_or_path

    rank = int(getattr(LiteLinear, "DEFAULT_RANK", 64))
    tag = _litelinear_patch_tag()
    config_filter_path = _litelinear_patch_config_path()
    config_filter_digest = "none"
    if config_filter_path is not None:
        if config_filter_path.is_file():
            config_filter_digest = hashlib.sha1(
                config_filter_path.read_bytes()).hexdigest()[:16]
        else:
            config_filter_digest = hashlib.sha1(
                str(config_filter_path).encode()).hexdigest()[:16]
    mode = "online" if _litelinear_online_patch_enabled() else "strict"
    dir_key = hashlib.sha1(
        f"{checkpoint_dir}|mode={mode}|rank={rank}|tag={tag}|cfg={config_filter_digest}".encode()
    ).hexdigest()[:16]
    cache_root = _litelinear_default_cache_root()
    patched_dir = cache_root / "patched_model_dirs" / dir_key
    patched_dir.mkdir(parents=True, exist_ok=True)
    if _litelinear_cached_patched_dir_ready(
            patched_dir,
            index_name=index_path.name,
            config_name=config_path.name,
            original_weight_map=weight_map):
        print(
            f"[LiteLinear] WanModel will reuse cached patched shards from {patched_dir}",
            flush=True,
        )
        return patched_dir

    resolve_shard = (
        _litelinear_resolve_or_build_patched_checkpoint
        if _litelinear_online_patch_enabled()
        else _litelinear_resolve_strict_checkpoint_path
    )
    safe_open_fn = _litelinear_safe_open_for_patch_build()
    shard_name_map = {}
    patched_weight_map = {}
    patched_count = 0
    for shard_name in sorted({str(name) for name in weight_map.values()}):
        src_shard = checkpoint_dir / shard_name
        resolved = resolve_shard(
            src_shard,
            rank=rank,
            tag=tag,
            cache_root=cache_root,
            safe_open_fn=safe_open_fn,
            log_prefix="[LiteLinear]",
            patch_config_path=config_filter_path,
            copy_original=_env_true("LITELINEAR_PATCH_COPY_ORIGINAL", "1"),
            force_rebuild=_env_true("LITELINEAR_PATCH_FORCE_REBUILD", "0"),
        ) if _litelinear_online_patch_enabled() else resolve_shard(
            src_shard,
            rank=rank,
            tag=tag,
            safe_open_fn=safe_open_fn,
            cache_root=cache_root,
            log_prefix="[LiteLinear]",
            patch_config_path=config_filter_path,
        )
        shard_name_map[shard_name] = resolved.name
        if resolved.name != shard_name:
            patched_count += 1
        _sync_file_reference(resolved, patched_dir / resolved.name)
        with safe_open_fn(str(resolved), framework="pt", device="cpu") as shard_file:
            for key in shard_file.keys():
                patched_weight_map[str(key)] = resolved.name

    patched_index = dict(data)
    patched_index["weight_map"] = patched_weight_map
    (patched_dir / index_path.name).write_text(
        json.dumps(patched_index, indent=2) + "\n",
        encoding="utf-8",
    )
    _sync_file_reference(config_path, patched_dir / config_path.name)

    if patched_count:
        print(
            f"[LiteLinear] WanModel will load patched shards from {patched_dir} "
            f"({patched_count}/{len(shard_name_map)} shard files remapped)",
            flush=True,
        )
        return patched_dir
    return pretrained_model_name_or_path


def _build_video_ffn_linear(in_features, out_features):
    if LiteLinear is None:
        warnings.warn(
            "lite_linear could not be imported for Wan video FFNs; "
            f"falling back to nn.Linear ({_LITELINEAR_IMPORT_ERROR}).",
            RuntimeWarning)
        return nn.Linear(in_features, out_features)
    # Use LiteLinear's native controls:
    # - LITELINEAR_DISABLED=1 to keep original linear weights active
    # - LiteLinear.DEFAULT_RANK to change the default decomposition rank
    return LiteLinear(in_features, out_features)


def assign_litelinear_module_keys(module):
    if LiteLinear is None:
        return 0

    assigned = 0
    for name, submodule in module.named_modules():
        if isinstance(submodule, LiteLinear):
            submodule._lite_key = name
            assigned += 1
    return assigned


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        # Keep LiteLinear restricted to the video FFN path only.
        # No replace_activation_proj_ needed here because Wan uses explicit
        # linear -> GELU -> linear, unlike wrapped activation-projection blocks.
        self.ffn = nn.Sequential(
            _build_video_ffn_linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            _build_video_ffn_linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs)
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(
                torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280))

    def forward(self, image_embeds):
        if hasattr(self, 'emb_pos'):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video) or 'vace'
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'flf2v', 'vace']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        if model_type == 'i2v' or model_type == 'flf2v':
            self.img_emb = MLPProj(1280, dim, flf_pos_emb=model_type == 'flf2v')

        # initialize weights
        self.init_weights()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        resolved = _resolve_litelinear_pretrained_dir(pretrained_model_name_or_path)
        patched = str(resolved) != str(pretrained_model_name_or_path)
        if patched:
            kwargs = dict(kwargs)
            # Patched LiteLinear shards can now use the low-memory loader path;
            # LiteLinear modules are finalized after load from the populated
            # factor buffers.
            kwargs.setdefault("low_cpu_mem_usage", True)
        model = super().from_pretrained(resolved, *model_args, **kwargs)
        if patched and LiteLinear is not None:
            finalized = LiteLinear.finalize_after_fast_load(model)
            if finalized:
                print(
                    f"[LiteLinear] Finalized {finalized} LiteLinear modules after fast load",
                    flush=True,
                )
        return model

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode or first-last-frame-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v' or self.model_type == 'flf2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 (x2) x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
