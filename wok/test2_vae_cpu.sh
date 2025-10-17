#!/usr/bin/bash
# Test 2: VAE CPU offloading only
echo "=== TEST 2: VAE offloading enabled (--vae_cpu) ==="
echo "Expected: Success - should save 100-200MB VRAM"
python ../generate.py --task t2v-1.3B --size 480*832 --ckpt_dir ./t2v-1.3b --offload_model True --vae_cpu --prompt "happy the dwarf and sneezy the dwarf wrestle to the death at madison square garden"
