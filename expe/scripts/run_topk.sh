#!/bin/bash

for topk in {6..8}; do

    python ablation.py --device_no 1 --gnn_type gin --pretrain_strategy supervised_contextpred --topk $topk


done