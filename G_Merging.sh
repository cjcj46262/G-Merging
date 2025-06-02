#!/usr/bin/env bash

python G_Merging.py --device_no=0 --gnn_type gin --pretrain_strategy contextpred
python G_Merging.py --device_no=0 --gnn_type gin --pretrain_strategy edgepred
python G_Merging.py --device_no=0 --gnn_type gcn --pretrain_strategy contextpred