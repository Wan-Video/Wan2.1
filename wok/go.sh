#!/usr/bin/bash
python ../generate.py  --task t2v-1.3B --size 480*832 --ckpt_dir ./t2v-1.3b --offload_model True --prompt "happy the dwarf and sneezy the dwarf wrestle to the death at madison square garden"
