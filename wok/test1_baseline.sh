#!/usr/bin/bash
# Test 1: Baseline (no flags) - expect OOM
echo "=== TEST 1: Baseline (no VAE offloading, no T5 offloading) ==="
echo "Expected: OOM Error during pipeline initialization"
python ../generate.py --task t2v-1.3B --size 480*832 --ckpt_dir ./t2v-1.3b --offload_model True --prompt "happy the dwarf and sneezy the dwarf wrestle to the death at madison square garden"
