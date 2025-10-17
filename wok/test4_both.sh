#!/usr/bin/bash
# Test 4: Both VAE and T5 CPU offloading
echo "=== TEST 4: Both VAE and T5 offloading enabled (--vae_cpu --t5_cpu) ==="
echo "Expected: Success - should save 150-250MB VRAM total"
python ../generate.py --task t2v-1.3B --size 480*832 --ckpt_dir ./t2v-1.3b --offload_model True --vae_cpu --t5_cpu --prompt "happy the dwarf and sneezy the dwarf wrestle to the death at madison square garden"
