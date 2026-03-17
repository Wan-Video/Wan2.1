# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.utils import _free_storage


def shard_model(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
):
    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype),
        device_id=device_id,
        sync_module_states=sync_module_states)
    return model


def shard_model_cpu_init(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=False,  # Key fix: disable sync_module_states for CPU init
):
    """
    Initialize FSDP with CPU-first approach to reduce GPU memory peak.
    """
    # Ensure model is on CPU and clear GPU cache
    model = model.cpu()
    torch.cuda.empty_cache()
    
    # Initialize FSDP with sync_module_states=False to allow CPU initialization
    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype),
        device_id=None,  # Keep on CPU during initialization
        sync_module_states=sync_module_states)  # Disabled for CPU init
    
    # Move to GPU after FSDP wrapping
    model = model.to(f"cuda:{device_id}")
    
    # Manual synchronization across ranks if needed
    if sync_module_states and torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    return model


def shard_model_with_cpu_offload(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
):
    """
    Initialize FSDP with CPU offload to minimize GPU memory usage.
    This keeps parameters on CPU and only loads them to GPU when needed.
    """
    from torch.distributed.fsdp import CPUOffload
    
    # Ensure model is on CPU and clear GPU cache aggressively
    model = model.cpu()
    torch.cuda.empty_cache()
    
    # Use CPU offload to minimize GPU memory usage
    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype),
        cpu_offload=CPUOffload(offload_params=True),  # Key: keep params on CPU
        device_id=device_id,
        sync_module_states=sync_module_states)
    
    return model

def free_model(model):
    for m in model.modules():
        if isinstance(m, FSDP):
            _free_storage(m._handle.flat_param.data)
    del model
    gc.collect()
    torch.cuda.empty_cache()