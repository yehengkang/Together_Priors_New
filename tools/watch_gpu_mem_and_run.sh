#!/bin/bash

THRESHOLD=1000   # MiB
INTERVAL=3600    # 1 hour

CMD="CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_yhk.py"

while true
do
    echo "[$(date)] Checking GPU memory usage..."

    MEMS=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

    all_idle=true
    idx=0

    for mem in $MEMS
    do
        echo "  GPU $idx memory used: ${mem} MiB"
        if [ "$mem" -ge "$THRESHOLD" ]; then
            all_idle=false
        fi
        idx=$((idx + 1))
    done

    if [ "$all_idle" = true ]; then
        echo "[$(date)] All GPUs memory < ${THRESHOLD} MiB. Launching training..."
        eval $CMD
        break
    else
        echo "[$(date)] GPUs not free. Sleep ${INTERVAL}s..."
        sleep $INTERVAL
    fi
done
