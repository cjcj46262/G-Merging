#!/bin/bash

for alpha in 1e-06 1e-05 1e-04 1e-03 1e-02 0.1 1 10; do

    python ablation_hyper.py --device_no 1 --gnn_type gin --pretrain_strategy supervised_contextpred --alpha $alpha

done

for alpha in 1e-06 1e-05 1e-04 1e-03 1e-02 0.1 1 10; do

    python ablation_hyper.py --device_no 1 --gnn_type gin --pretrain_strategy supervised_edgepred --alpha $alpha

done

for alpha in 1e-06 1e-05 1e-04 1e-03 1e-02 0.1 1 10; do

    python ablation_hyper.py --device_no 1 --gnn_type gcn --pretrain_strategy supervised_contextpred --alpha $alpha

done

for lamweight in 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.225 0.25; do

    python ablation_hyper.py --device_no 1 --gnn_type gin --pretrain_strategy supervised_contextpred --lam --lamweight $lamweight

done

for lamweight in 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.225 0.25; do

    python ablation_hyper.py --device_no 1 --gnn_type gin --pretrain_strategy supervised_edgepred --lam --lamweight $lamweight

done

for lamweight in 0.05 0.075 0.1 0.125 0.15 0.175 0.2 0.225 0.25; do

    python ablation_hyper.py --device_no 1 --gnn_type gcn --pretrain_strategy supervised_contextpred --lam --lamweight $lamweight

done