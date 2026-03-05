#!/usr/bin/bash
# Test 3: T5 CPU offloading only
echo "=== TEST 3: T5 offloading enabled (--t5_cpu) ==="
echo "Expected: Unknown - might still OOM, depends on T5 memory footprint"
python ../generate.py --task t2v-1.3B --size 480*832 --ckpt_dir ./t2v-1.3b --offload_model True --t5_cpu --prompt "happy the dwarf and sneezy the dwarf wrestle to the death at madison square garden"
