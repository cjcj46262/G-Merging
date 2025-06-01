# G-Merging

Code of Paper: "G-Merging: Graph Models Merging for Parameter-Efficient Multi-Task Knowledge Consolidation"

**Keywords**: Model Merging, Parameter Efficient Fine-Tuning, Multi-task Learning

## Abstract

The pretrain-finetuning paradigm has achieved notable success in graph learning. Moreover, merging models fine-tuned on different tasks to enable a parameter-efficient model with multi-task capabilities is gaining increasing attention for its practicality. However, existing model merging methods, such as weight averaging and task arithmetic, struggle to generalize well to graph structures and Graph Neural Network (GNN) models due to the unique structural heterogeneity of graph data. In this paper, we propose an innovative graph model merging framework called G-Merging for merging multiple task-specific fine-tuned GNN models. G-Merging first employs task arithmetic to coarsely merge graph models, capturing shared cross-task knowledge. Second, it introduces a Topology-aware Wasserstein Distance (TWD) loss to train lightweight task adapters, preserving domain-specific graph patterns via aligning the embeddings of merged and fine-tuned models. Third, G-Merging integrates the adapters into a training-free, topology-aware router within a mixture-of-experts (MoE) architecture, dynamically routing input graphs to task-specific adapters based on structural similarity, thereby mitigating conflicts and enhancing knowledge sharing. Extensive experiments on 8 graph downstream datasets demonstrate the effectiveness of G-Merging, showing impressive performance close to or exceeding individual finetuned models while improving parameters and training efficiency.

## Framework

![overall framework](./G_Merging.png)

[//]: # (<p align="center">)

[//]: # (  <img src="https://github.com/kimsu55/ToxExpert/blob/main/img/fig3_main_arch.jpg" width="500" title="The overall framework of TopExpert">)

[//]: # (</p>)

## Dependency
We used Python 3.10.16, PyTorch 2.5.1, PyTorch Geometric 2.6.1. For the Python packages, please see requirements.txt.

```
numpy==2.1.2
pandas==2.2.3
pillow==11.0.0
scipy==1.15.2
torch==2.5.1+cu121
torch-geometric==2.6.1
torch_cluster==1.6.3+pt25cu121
torch_scatter==2.1.2+pt25cu121
torch_sparse==0.6.18+pt25cu121
torch_spline_conv==1.2.2+pt25cu121
torchaudio==2.5.1+cu121
torchvision==0.20.1+cu121
```
## Download models
we provide the .pth of pre-trained GNN models and full fine-tuned models,

You can make a directory `./data` and download all the models through the Google drive from 

Make sure the data files are in the `./models` folder:
```
project
│   README.md
│   data_process.py
│   GCAL.py
|   main.py
|   ...
|
└───data
│       │   twitch
|       |   fb100
│       │   elliptic
│       │   ogbn-arxiv
```

The detailed data processing is in the `data_process.py`.

## Run the code
Simply run the following command to get started.
```
python G_Merging.py --device_no=0 --gnn_type gin --pretrain_strategy contextpred
python G_Merging.py --device_no=0 --gnn_type gin --pretrain_strategy edgepred
python G_Merging.py --device_no=0 --gnn_type gcn --pretrain_strategy contextpred
```



