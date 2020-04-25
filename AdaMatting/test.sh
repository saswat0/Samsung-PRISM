#/bin/bash

python main.py \
    --mode=test \
    --valid_portion=5 \
    --write_log \
    --cuda \
    --gpu=0
