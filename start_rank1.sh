#!/bin/bash

# Set environment variables
export PATH=/data/apps/wan-21/wan-env/bin:$PATH
export MASTER_ADDR='10.79.79.197'
export MASTER_PORT='7860'
export RANK='1'
export WORLD_SIZE='2'
export LOCAL_RANK='0'
export NCCL_SOCKET_IFNAME='enp33s0'

# Run the generate.py script with specified parameters
# like: ./start_rank1.sh --dit_fsdp --sample_steps 40 --task i2v-14B --size 480*640 --ckpt_dir ./Wan2.1-I2V-14B-480P --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
python generate.py "$@"
