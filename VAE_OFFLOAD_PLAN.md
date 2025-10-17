# VAE Offloading Implementation & Testing Plan

## Overview
Add `--vae_cpu` flag to enable VAE offloading to save ~100-200MB VRAM during text-to-video generation.

## Implementation Plan

### Phase 1: Code Changes

**1. Add `--vae_cpu` flag to generate.py**
- Add argument to parser (similar to `--t5_cpu`)
- Default: `False` (maintain current upstream behavior)
- Pass to pipeline constructors
- Independent flag (works regardless of `offload_model` setting)

**2. Update Pipeline Constructors**
- Add `vae_cpu` parameter to `__init__` methods in:
  - `WanT2V` (text2video.py)
  - `WanI2V` (image2video.py)
  - `WanFLF2V` (first_last_frame2video.py)
  - `WanVace` (vace.py)

**3. Conditional VAE Initialization**
- If `vae_cpu=True`: Initialize VAE on CPU
- If `vae_cpu=False`: Initialize VAE on GPU (current behavior)

**4. Update Offload Logic**
- Only move VAE to/from GPU when `vae_cpu=True`
- When `vae_cpu=False`, VAE stays on GPU (no extra transfers)

## Phase 2: Testing Plan

### Test Scripts to Create:

```bash
# wok/test1_baseline.sh - No flags (expect OOM)
python ../generate.py --task t2v-1.3B --size 480*832 --ckpt_dir ./t2v-1.3b --offload_model True --prompt "..."

# wok/test2_vae_cpu.sh - Only VAE offloading
python ../generate.py --task t2v-1.3B --size 480*832 --ckpt_dir ./t2v-1.3b --offload_model True --vae_cpu --prompt "..."

# wok/test3_t5_cpu.sh - Only T5 offloading
python ../generate.py --task t2v-1.3B --size 480*832 --ckpt_dir ./t2v-1.3b --offload_model True --t5_cpu --prompt "..."

# wok/test4_both.sh - Both flags
python ../generate.py --task t2v-1.3B --size 480*832 --ckpt_dir ./t2v-1.3b --offload_model True --vae_cpu --t5_cpu --prompt "..."
```

### Expected Results:

| Test | Flags | Expected Outcome | Memory Peak |
|------|-------|------------------|-------------|
| 1 | None | ❌ OOM Error | ~VRAM_MAX + 100MB |
| 2 | `--vae_cpu` | ✅ Success | ~VRAM_MAX - 100-200MB |
| 3 | `--t5_cpu` | ? (might still OOM) | ~VRAM_MAX - 50MB |
| 4 | `--vae_cpu --t5_cpu` | ✅ Success | ~VRAM_MAX - 150-250MB |

### Actual Test Results:

**Hardware:** 11.49 GiB VRAM GPU

| Test | Flags | Actual Outcome | Notes |
|------|-------|----------------|-------|
| 1 | None | ❌ OOM Error | Failed trying to allocate 80MB, only 85.38MB free |
| 2 | `--vae_cpu` | ✅ Success | Completed successfully after fixes |
| 3 | `--t5_cpu` | ✅ Success | No OOM, completed successfully |
| 4 | `--vae_cpu --t5_cpu` | ✅ Success | Completed with maximum VRAM savings |

**Key Findings:**
- Baseline OOM occurred when trying to move T5 to GPU with DiT already loaded
- VAE offloading alone is sufficient to fix the OOM
- T5 offloading alone is also sufficient (surprising but effective!)
- Both flags together provide maximum VRAM savings for users with limited GPU memory
- All approaches work by freeing VRAM at critical moments during the pipeline execution

**Conclusion:**
The `--vae_cpu` flag is a valuable addition for consumer GPU users, complementing the existing `--t5_cpu` optimization and following the same design pattern.

## Phase 3: Documentation & PR

### 1. Results Document
- Memory usage for each test
- Performance impact (if any) from CPU↔GPU transfers
- Recommendations for users

### 2. PR Components
- Feature description
- Memory savings benchmarks
- Backward compatible (default=False)
- Use cases: when to enable `--vae_cpu`

## Design Decisions

1. **Independence**: `vae_cpu` works independently of `offload_model` flag (mirrors `t5_cpu` behavior)
2. **Default False**: Maintains current upstream behavior for backward compatibility
3. **Conditional Transfers**: Only add GPU↔CPU transfers when flag is enabled

## Memory Analysis

**Current Pipeline Memory Timeline:**
```
Init:    [T5-CPU] [VAE-GPU] [DiT-GPU]  <- OOM here during init!
Encode:  [T5-GPU] [VAE-GPU] [DiT-GPU]
Loop:    [T5-CPU] [VAE-GPU] [DiT-GPU]  <- VAE not needed but wasting VRAM
Decode:  [T5-CPU] [VAE-GPU] [DiT-CPU]  <- Only now is VAE actually used
```

**With `--vae_cpu` Enabled:**
```
Init:    [T5-CPU] [VAE-CPU] [DiT-GPU]  <- VAE no longer occupying VRAM
Encode:  [T5-GPU] [VAE-CPU] [DiT-GPU]
Loop:    [T5-CPU] [VAE-CPU] [DiT-GPU]  <- VAE stays on CPU during loop
Decode:  [T5-CPU] [VAE-GPU] [DiT-CPU]  <- VAE moved to GPU only for decode
```

## Implementation Details

### Critical Fixes Applied:

1. **DiT Offloading Before T5 Load** (when `offload_model=True` and `t5_cpu=False`)
   - DiT must be offloaded to CPU before loading T5 to GPU
   - Otherwise T5 allocation fails with OOM
   - Added automatic DiT→CPU before T5→GPU transition

2. **VAE Scale Tensors** (when `vae_cpu=True`)
   - VAE wrapper class stores `mean` and `std` tensors separately
   - These don't move with `.model.to(device)`
   - Must explicitly move scale tensors along with model
   - Fixed in all encode/decode operations

3. **Conditional Offloading Logic**
   - VAE offloading only triggers when `vae_cpu=True`
   - Works independently of `offload_model` flag
   - Mirrors `t5_cpu` behavior for consistency

## Files Modified

1. `generate.py` - Add argument parser
2. `wan/text2video.py` - WanT2V pipeline
3. `wan/image2video.py` - WanI2V pipeline
4. `wan/first_last_frame2video.py` - WanFLF2V pipeline
5. `wan/vace.py` - WanVace pipeline
6. `wok/test*.sh` - Test scripts
