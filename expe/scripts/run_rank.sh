#!/bin/bash

for rank in 10 20 30 40 50 ; do

    python ablation_rank.py --device_no 1 --gnn_type gin --pretrain_strategy supervised_contextpred --rank $rank 


done