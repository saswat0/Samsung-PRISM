#/bin/bash

python main.py \
    --mode=train \
    --valid_portion=5 \
    --batch_size=16 \
    --epochs=120 \
    --lr=0.0001 \
    --decay_iters=51182 \
    --cuda \
    --gpu=0,1,2,3 \
    --write_log \
    --save_ckpt \
    # --resume=./ckpts/ckpt.tar
