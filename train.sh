#!/bin/bash

python seq2seq_attention.py --mode=train --article_key=content --abstract_key=title --num_gpus=0 --truncate_input=True --data_path=../data/data --log_root=../log_root --vocab_path=../data/vocab --train_dir=../log_root/train

