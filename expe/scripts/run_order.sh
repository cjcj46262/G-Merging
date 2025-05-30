#!/bin/bash

for order in 0 1 2 3 ; do

    python ablation_order.py --device_no 0 --gnn_type gin --pretrain_strategy supervised_contextpred --gtot_order $order


done

