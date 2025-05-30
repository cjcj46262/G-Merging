import math
import argparse
import sys
import re
import os
import contextlib
import copy
import time

import matplotlib.pyplot as plt
import seaborn as sns


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from splitters import data_split
from loader import MoleculeDataset
from model import GNN_topexpert, GNN_graphpred
from emr_merge import emr_merge, apply_vector, weight_average, weight_average_train
from task_vectors import TaskVector
from surgeryV1 import AlphaWrapper_Surgery
from MWD.gtot_tuning import GTOTRegularization
from MWD.delta import IntermediateLayerGetter, L2Regularization, FrobeniusRegularization
from util import *
from collections import OrderedDict
from itertools import chain

criterion = nn.BCEWithLogitsLoss(reduction="none")


# class BinaryClassificationEntropy(nn.Module):
#     def __init__(self):
#         super(BinaryClassificationEntropy, self).__init__()

#     def forward(self, logits):
#         probabilities = torch.sigmoid(logits)
#         entropy = -probabilities * torch.log(probabilities) - (1 - probabilities) * torch.log(1 - probabilities)
        
#         return entropy

# criterion1 = BinaryClassificationEntropy()

# def kl_divergence_loss(output_probs, prior_probs):
#     # print(output_probs)
#     # print(torch.log(prior_probs))
#     # print(torch.log(output_probs))
#     kl_loss = output_probs * (torch.log(output_probs) - torch.log(prior_probs))
#     return kl_loss.mean()





def load_args():
    parser = argparse.ArgumentParser()

# seed & device
    parser.add_argument('--device_no', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting the dataset.")
   
#dataset
    parser.add_argument('--dataset_dir', type=str, default='./data', help='directory of dataset')
    parser.add_argument('--dataset', type=str, default='bbbp', help='root directory of dataset')
    parser.add_argument('--split', type=str, default="scaffold", help="random or scaffold or random_scaffold")

#model
    parser.add_argument('-i', '--input_model_file', type=str, default='', help='filename to read the model (if there is any)')
    parser.add_argument('-c', '--ckpt_all', type=str, default='',
                        help='filename to read the model ')
    

    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max, concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--pretrain_strategy', type=str, default="supervised_contextpred")
    parser.add_argument('--rank', type=int, default=30)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--topk', type=int, default=8)


# train
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')


#optimizer
    parser.add_argument('--lr_ada', type=float, default=1e-3,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_graph', type=float, default=1e-3,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_node', type=float, default=1e-3,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')

## loss balance
    parser.add_argument('--alpha', type=float, default=1, help="balance parameter for TWD and L1")

## clustering
    parser.add_argument('--min_temp', type=float, default=1, help=" temperature for gumble softmax, annealing")
    parser.add_argument('--num_experts', type=int, default=8)
    parser.add_argument('--gate_dim', type=int, default=50, help="gate embedding space dimension, 50 or 300")

## GTOT
    parser.add_argument('--gtot_order', default=1, type=int, help='A^{k} in graph topology OT')


    
    args = parser.parse_args()
    args.device = torch.device("cuda:" + str(args.device_no)) if torch.cuda.is_available() else torch.device("cpu")

    # Bunch of classification tasks
    if args.dataset == "tox21":
        args.num_tasks = 12
        args.num_classes = 2
    elif args.dataset == "hiv":
        args.num_tasks = 1
        args.num_classes = 2
    elif args.dataset == "pcba":
        args.num_tasks = 128
        args.num_classes = 2
    elif args.dataset == "muv":
        args.num_tasks = 17
        args.num_classes = 2
    elif args.dataset == "bace":
        args.num_tasks = 1
        args.num_classes = 2
    elif args.dataset == "bbbp":
        args.num_tasks = 1
        args.num_classes = 2
    elif args.dataset == "toxcast":
        args.num_tasks = 617
        args.num_classes = 2
    elif args.dataset == "sider":
        args.num_tasks = 27
        args.num_classes = 2
    elif args.dataset == "clintox":
        args.num_tasks = 2
        args.num_classes = 2
    else:
        raise ValueError("Invalid dataset name.")

    return args


def train(args, model, finetune_model, loader, optimizer, loss_func, finetune_getter, target_getter1, target_getter2, backbone_regularization=None):
    # alpha_model.train()
    # finetune_model.eval()
    all_fea_loss = 0.
    all_gtot_loss = 0.
    
    for batch in loader:
        batch = batch.to(args.device)

        intermediate_output_s, feature_s = finetune_getter(batch.x, batch.edge_index, batch.edge_attr, batch.batch)  # batch.batch is a column vector which maps each node to its respective graph in the batch
        intermediate_output_t1, feature_t = target_getter1(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        intermediate_output_t2, feature_t = target_getter2(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        intermediate_output_t = OrderedDict()
        if args.gnn_type == 'gin':
            for i in range(args.num_layer):
                intermediate_output_t[f'gnn.gnns.{i}.mlp.2'] = intermediate_output_t1[f'gnn.gnns.{i}.mlp.2'] - intermediate_output_t2[f'gnn.surgery_mlps.{i}.2']
        if args.gnn_type == 'gcn':
            for i in range(args.num_layer):
                intermediate_output_t[f'gnn.gnns.{i}'] = intermediate_output_t1[f'gnn.gnns.{i}'] - intermediate_output_t2[f'gnn.surgery_mlps.{i}.2']

        # print(intermediate_output_s.values())

        loss_reg_backbone = backbone_regularization(intermediate_output_s, intermediate_output_t, batch)
        # print(intermediate_output_s['gnn.gnns.0.mlp.2'].shape)
        # print(intermediate_output_t['gnn.gnns.0.mlp.2'].shape)
        
        # surgery_l2_loss = surgery_regularization()


        # surgery_features = model.pool(model.gnn(batch.x, batch.edge_index, batch.edge_attr), batch.batch)
        # finetuned_features = finetune_model.pool(finetune_model.gnn(batch.x, batch.edge_index, batch.edge_attr), batch.batch)

        # alllayers_intermedie_output_s = torch.cat([intermediate_output_s[f'gnn.gnns.{i}.mlp.2'] for i in range(args.num_layer)],dim=0)
        # alllayers_intermedie_output_t = torch.cat([intermediate_output_t[f'gnn.gnns.{i}.mlp.2'] for i in range(args.num_layer)],dim=0)
        # print(alllayers_intermedie_output_s.shape)
        # print(output_s.shape)



        feature_loss = loss_func(feature_s, feature_t)
        # print(feature_loss)
        # print(f'feature_loss: {feature_loss}')

        # sys.exit()

        # loss = feature_loss
        # loss = loss_reg_backbone
        loss = loss_reg_backbone + args.alpha * feature_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_gtot_loss += loss_reg_backbone.item()
        all_fea_loss += feature_loss.item()
    
    return all_gtot_loss / len(loader), all_fea_loss / len(loader)



def eval(args, model, loader):
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()  # 临时启用训练模式
    
    moe_weights = []
    y_true, y_scores = [], []
    for batch in loader:
        batch = batch.to(args.device)

        with torch.no_grad():
            #clf_logit, z, q_origin = model(batch)
            #q, q_idx = model.assign_head(q_origin) # N x tasks x head
            #scores = torch.sum(torch.sigmoid(clf_logit) * q, dim=-1)          
            logits,_,moe_weight = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)   
            #rep = model.get_graph_rep(batch)
            # finetuned_features = finetune_model.pool(finetune_model.gnn(batch.x, batch.edge_index, batch.edge_attr), batch.batch)
            # features[:,150:] = finetuned_features[:,150:]
            # logits = model.model.graph_pred_linear(features)
            scores = torch.sigmoid(logits)
            #print(scores)
            #print(torch.isnan(scores))
        y_true.append(batch.y.view(batch.id.shape[0], -1))
        y_scores.append(scores)    
        moe_weights.append(moe_weight)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
    moe_weights = torch.cat(moe_weights, dim=0).mean(0).cpu().numpy()
    avg_roc = cal_roc(y_true, y_scores)

    return  avg_roc, moe_weights



def test_one_dataset(args):
    set_seed(args.seed)    
    model_file = f'/public24_data/cj/CcccJjjj/ftmodels/{args.gnn_type}_{args.pretrain_strategy}/{args.gnn_type}_{args.dataset}_sd0.pt'
    print(model_file)
    dict_m = torch.load(model_file, map_location='cpu')
    dict_para = dict_m['model_state_dict']
    # print(torch.zeros(300))
    # sys.exit()
    # dict_para['gnn.batch_norms.0.running_mean'] = torch.zeros(300)
    #print(dict_para.keys())
    #sys.exit()

    # dataset split & data loader  supervised_
    dataset = MoleculeDataset(args.dataset_dir + "/" + args.dataset, dataset=args.dataset)
    train_dataset, valid_dataset, test_dataset = data_split(args, dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model = GNN_graphpred(args, moe=True)
    # finetune_model = GNN_graphpred(args)
    model.load_state_dict(dict_para, strict=False)
    # finetune_model.load_state_dict(dict_para)


    exam_datasets = ['tox21', 'toxcast', 'sider', 'clintox', 'bbbp', 'bace', 'hiv', 'muv']
    if args.gnn_type == 'gin':
        pretrained_path = f'/public24_data/cj/CcccJjjj/model_gin/{args.pretrain_strategy}.pth'
    else:
        pretrained_path = f'/public24_data/cj/CcccJjjj/model_architecture/{args.gnn_type}_{args.pretrain_strategy}.pth'
    task_vectors = [
    TaskVector(pretrained_path, f'/public24_data/cj/CcccJjjj/ftmodels/{args.gnn_type}_{args.pretrain_strategy}/{args.gnn_type}_{dataset_name}_sd0.pt') for dataset_name in exam_datasets
    ]
    task_vector_sum = sum(task_vectors)
    if args.gnn_type == 'gin' and args.pretrain_strategy == 'supervised_contextpred':
        scaling_coef_ = 0.2
    elif args.gnn_type == 'gin' and args.pretrain_strategy == 'supervised_edgepred':
        scaling_coef_ = 0.175
    elif args.gnn_type == 'gcn' and args.pretrain_strategy == 'supervised_contextpred':
        scaling_coef_ = 0.11
    

    vector_unified, masks, rescalers = emr_merge(task_vectors)
    # masks_list = torch.zeros(8)
    # print(rescalers)

    ind = exam_datasets.index(args.dataset)
    task_vector_recon = {}
    for n in vector_unified:
        task_vector_recon[n] =  vector_unified[n] * masks[n][ind] * rescalers[ind]
    # for i in range(8):
    #     for n in vector_unified:
    #         masks_list[i] += torch.sum(masks[n][i])
    # print(masks_list)
    # sys.exit()
    # print(rescalers)
    # print(masks['gnns.0.mlp.0.weight'][ind])

    pretrained_state_dict = torch.load(pretrained_path, map_location='cpu')
    task_params = {}
    for key in pretrained_state_dict:
        if key not in task_vector_recon:
            print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
            continue
            task_params[key] = pretrained_state_dict[key]
            task_params1[key] = pretrained_state_dict[key]      
        else:

            task_params[key] = pretrained_state_dict[key] + scaling_coef_ * task_vector_sum.vector[key]



    model.gnn.load_state_dict(task_params, strict=False)
    # for module in model.modules():
    #     print(module)
    # sys.exit()


    # return_layers = [f'gnn.gnns.{i}.mlp.2' for i in range(args.num_layer)]
    # return_layers_sur = [f'gnn.surgery_mlps.{i}.2' for i in range(args.num_layer)]

    # finetune_getter = IntermediateLayerGetter(finetune_model, return_layers=return_layers)
    # target_getter1 = IntermediateLayerGetter(model, return_layers=return_layers)
    # target_getter2 = IntermediateLayerGetter(model, return_layers=return_layers_sur)
    # backbone_regularization = GTOTRegularization(order=args.gtot_order, args=args)
    # # surgery_regularization = L2Regularization(nn.ModuleList([model.gnn.surgery_mlps]))


    # # alpha_model = AlphaWrapper_Surgery(model, args)
    # # alpha_model.to(args.device)
    # model.to(args.device)
    # finetune_model.to(args.device)

    # optimizer = torch.optim.Adam(
    #     [
    #         {"params": model.surgery_mlp.parameters(), "lr": args.lr_graph},
    #         {"params": model.gnn.surgery_mlps.parameters(), "lr": args.lr_node},
    #     ],
    #     betas=(0.9, 0.999),
    #     weight_decay=0.
    # )
    # # optimizer = torch.optim.Adam(alpha_model.collect_trainable_params(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.)
    # loss_func = torch.nn.L1Loss()
    # # loss_func = torch.nn.MSELoss()

    
    # for epoch in range(args.epochs):
    #     gtot_loss, fea_loss = train(args, model, finetune_model, train_loader, optimizer, loss_func, finetune_getter, target_getter1, target_getter2, backbone_regularization)
    #     print(f'epoch {epoch} finished, gtot loss: {gtot_loss}, fea loss: {fea_loss}')


    list_datasets = ['tox21', 'toxcast', 'sider', 'clintox', 'bbbp', 'bace', 'hiv', 'muv']
    for i, d_name in enumerate(list_datasets):
        model_file = f'./ablation/{args.gnn_type}_{args.pretrain_strategy}/AGMM/{d_name}_aligners.pth'
        dict_moe = torch.load(model_file, map_location='cpu')
        # print(dict_moe['aligner_layer0'].keys())
        # print(model.surgery_moe[i].state_dict().keys())
        # sys.exit()
        for layer in range(args.num_layer):
            model.gnn.surgery_moe_layers[layer][i].load_state_dict(dict_moe[f'aligner_layer{layer}'])
        model.surgery_moe[i].load_state_dict(dict_moe['aligner_graph'])
    model = model.to(args.device)





    te_acc, moe_weights = eval(args, model, test_loader)
    # checkpoint = {
    #     'aligner_layer0': model.gnn.surgery_mlps[0].state_dict(),
    #     'aligner_layer1': model.gnn.surgery_mlps[1].state_dict(),
    #     'aligner_layer2': model.gnn.surgery_mlps[2].state_dict(),
    #     'aligner_layer3': model.gnn.surgery_mlps[3].state_dict(),
    #     'aligner_layer4': model.gnn.surgery_mlps[4].state_dict(),
    #     'aligner_graph': model.surgery_mlp.state_dict(),
        
    # }
    # os.makedirs(f'./shell/{args.gnn_type}_{args.pretrain_strategy}', exist_ok=True)
    # torch.save(checkpoint, f"./shell/{args.gnn_type}_{args.pretrain_strategy}/taskArith_surgeryV2_GTOT_218/{args.dataset}_aligners.pth")
    # print(f'test acc:{te_acc:.2f} ')
    # print(moe_weights.type)
    # sys.exit()
    # print(te_acc)
    return te_acc, moe_weights



def train_one_adapters(args):
    set_seed(args.seed)    
    model_file = f'/public24_data/cj/CcccJjjj/ftmodels/{args.gnn_type}_{args.pretrain_strategy}/{args.gnn_type}_{args.dataset}_sd0.pt'
    print(model_file)
    dict_m = torch.load(model_file, map_location='cpu')
    dict_para = dict_m['model_state_dict']

    # dataset split & data loader  supervised_
    dataset = MoleculeDataset(args.dataset_dir + "/" + args.dataset, dataset=args.dataset)
    train_dataset, valid_dataset, test_dataset = data_split(args, dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model = GNN_graphpred(args, surgery=True)
    finetune_model = GNN_graphpred(args)
    model.load_state_dict(dict_para, strict=False)
    finetune_model.load_state_dict(dict_para)


    exam_datasets = ['tox21', 'toxcast', 'sider', 'clintox', 'bbbp', 'bace', 'hiv', 'muv']
    if args.gnn_type == 'gin':
        pretrained_path = f'/public24_data/cj/CcccJjjj/model_gin/{args.pretrain_strategy}.pth'
    else:
        pretrained_path = f'/public24_data/cj/CcccJjjj/model_architecture/{args.gnn_type}_{args.pretrain_strategy}.pth'
    task_vectors = [
    TaskVector(pretrained_path, f'/public24_data/cj/CcccJjjj/ftmodels/{args.gnn_type}_{args.pretrain_strategy}/{args.gnn_type}_{dataset_name}_sd0.pt') for dataset_name in exam_datasets
    ]
    task_vector_sum = sum(task_vectors)
    if args.gnn_type == 'gin' and args.pretrain_strategy == 'supervised_contextpred':
        scaling_coef_ = 0.2
    elif args.gnn_type == 'gin' and args.pretrain_strategy == 'supervised_edgepred':
        scaling_coef_ = 0.175
    elif args.gnn_type == 'gcn' and args.pretrain_strategy == 'supervised_contextpred':
        scaling_coef_ = 0.11


    vector_unified, masks, rescalers = emr_merge(task_vectors)
    # masks_list = torch.zeros(8)
    # print(rescalers)

    ind = exam_datasets.index(args.dataset)
    task_vector_recon = {}
    for n in vector_unified:
        task_vector_recon[n] =  vector_unified[n] * masks[n][ind] * rescalers[ind]
    # for i in range(8):
    #     for n in vector_unified:
    #         masks_list[i] += torch.sum(masks[n][i])
    # print(masks_list)
    # sys.exit()
    # print(rescalers)
    # print(masks['gnns.0.mlp.0.weight'][ind])

    pretrained_state_dict = torch.load(pretrained_path, map_location='cpu')
    task_params = {}
    for key in pretrained_state_dict:
        if key not in task_vector_recon:
            print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
            continue
            task_params[key] = pretrained_state_dict[key]
            task_params1[key] = pretrained_state_dict[key]      
        else:

            task_params[key] = pretrained_state_dict[key] + scaling_coef_ * task_vector_sum.vector[key]



    model.gnn.load_state_dict(task_params, strict=False)
    # for module in model.modules():
    #     print(module)
    # sys.exit()


    if args.gnn_type == 'gin':
        return_layers = [f'gnn.gnns.{i}.mlp.2' for i in range(args.num_layer)]
    if args.gnn_type == 'gcn':
        return_layers = [f'gnn.gnns.{i}' for i in range(args.num_layer)]
    return_layers_sur = [f'gnn.surgery_mlps.{i}.2' for i in range(args.num_layer)]

    finetune_getter = IntermediateLayerGetter(finetune_model, return_layers=return_layers)
    target_getter1 = IntermediateLayerGetter(model, return_layers=return_layers)
    target_getter2 = IntermediateLayerGetter(model, return_layers=return_layers_sur)
    backbone_regularization = GTOTRegularization(order=args.gtot_order, args=args)
    # surgery_regularization = L2Regularization(nn.ModuleList([model.gnn.surgery_mlps]))


    # alpha_model = AlphaWrapper_Surgery(model, args)
    # alpha_model.to(args.device)
    model.to(args.device)
    finetune_model.to(args.device)

    optimizer = torch.optim.Adam(
        [
            {"params": model.surgery_mlp.parameters(), "lr": args.lr_graph},
            {"params": model.gnn.surgery_mlps.parameters(), "lr": args.lr_node},
        ],
        betas=(0.9, 0.999),
        weight_decay=0.
    )
    # optimizer = torch.optim.Adam(alpha_model.collect_trainable_params(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.)
    loss_func = torch.nn.L1Loss()
    # loss_func = torch.nn.MSELoss()

    
    for epoch in range(args.epochs):
        gtot_loss, fea_loss = train(args, model, finetune_model, train_loader, optimizer, loss_func, finetune_getter, target_getter1, target_getter2, backbone_regularization)
        print(f'epoch {epoch} finished, gtot loss: {gtot_loss}, fea loss: {fea_loss}')


    # list_datasets = ['tox21', 'toxcast', 'sider', 'clintox', 'bbbp', 'bace', 'hiv', 'muv']
    # for i, d_name in enumerate(list_datasets):
    #     model_file = f'./shell/contextpred/taskArith_surgeryV2_GTOT/{d_name}_aligners.pth'
    #     dict_moe = torch.load(model_file, map_location='cpu')
    #     # print(dict_moe['aligner_layer0'].keys())
    #     # print(model.surgery_moe[i].state_dict().keys())
    #     # sys.exit()
    #     for layer in range(args.num_layer):
    #         model.gnn.surgery_moe_layers[layer][i].load_state_dict(dict_moe[f'aligner_layer{layer}'])
    #     model.surgery_moe[i].load_state_dict(dict_moe['aligner_graph'])
    # model = model.to(args.device)





    te_acc = eval(args, model, test_loader)
    checkpoint = {
        'aligner_layer0': model.gnn.surgery_mlps[0].state_dict(),
        'aligner_layer1': model.gnn.surgery_mlps[1].state_dict(),
        'aligner_layer2': model.gnn.surgery_mlps[2].state_dict(),
        'aligner_layer3': model.gnn.surgery_mlps[3].state_dict(),
        'aligner_layer4': model.gnn.surgery_mlps[4].state_dict(),
        'aligner_graph': model.surgery_mlp.state_dict(),
        
    }
    os.makedirs(f'./ablation/{args.gnn_type}_{args.pretrain_strategy}/AGMM_wo_graph_adapter', exist_ok=True)
    torch.save(checkpoint, f"./ablation/{args.gnn_type}_{args.pretrain_strategy}/AGMM_wo_graph_adapter/{args.dataset}_aligners.pth")
    print(f'test acc:{te_acc:.2f} ')
    # print(te_acc)
    return te_acc


def main(args):
    # all_acc = joint_train_datasets(args)

    list_datasets = ['tox21', 'toxcast', 'sider', 'clintox', 'bbbp', 'bace', 'hiv', 'muv']
    list_num_tasks = [12, 617, 27, 2, 1, 1, 1, 17]
    list_batch_sizes = [512, 512, 16, 256, 512, 512, 64, 256]
    list_alpha = [1,1,1,1,1,1,1,1]
    list_lr_graph = [1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3]
    list_lr_node = [1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3]

    # start_time = time.time()

    # for index, (dataset_name, num_tasks, batch_size, alpha, lr_graph, lr_node) in enumerate(zip(list_datasets, list_num_tasks, list_batch_sizes, list_alpha, list_lr_graph, list_lr_node)):
    #     args.index = index
    #     args.dataset = dataset_name
    #     args.num_tasks = num_tasks
    #     args.batch_size = batch_size
    #     args.alpha = alpha
    #     args.lr_graph = lr_graph
    #     args.lr_node = lr_node
    #     acc = train_one_adapters(args)
    
    
    # end_time = time.time() 
    # elapsed_time = end_time - start_time
    # minutes = int(elapsed_time // 60)
    # seconds = int(elapsed_time % 60)
    # print(f"train time : {elapsed_time:.2f} s")
    # print(f"train time : {minutes:.2f} min {seconds:.2f} s")


    # all_acc = []
    # weights_list = []
    # for index, (dataset_name, num_tasks, batch_size, alpha, lr_graph, lr_node) in enumerate(zip(list_datasets, list_num_tasks, list_batch_sizes, list_alpha, list_lr_graph, list_lr_node)):
    #     args.index = index
    #     args.dataset = dataset_name
    #     args.num_tasks = num_tasks
    #     args.batch_size = batch_size
    #     args.alpha = alpha
    #     args.lr_graph = lr_graph
    #     args.lr_node = lr_node
    #     # acc = train_one_adapters(args)
    #     acc, weights = test_one_dataset(args)
    #     weights_list.append(weights)
    #     all_acc.append(acc)
    #     # with open(f'./ablation/{args.gnn_type}_{args.pretrain_strategy}/AGMM_top{args.topk}.txt', 'a') as file:
    #     #     file.write(f'data name: {dataset_name}\n')
    #     #     file.write(f'Test ROC AUC score: {acc}\n')
    # # with open(f'./ablation/{args.gnn_type}_{args.pretrain_strategy}/AGMM_top2.txt', 'a') as file:
    # #     file.write(f"train time : {minutes:.2f} min {seconds:.2f} s\n")
    # #     file.write(f"train time : {elapsed_time:.2f} s")
    # weights_np = np.stack(weights_list, axis=0)

    weights_np = np.array([
        [0.3137147, 0.11687851, 0.1335685, 0.11932724, 0.0740407, 0.08762258, 0.07789325, 0.07695454],
        [0.11489082, 0.319199, 0.14032403, 0.1131031, 0.06935998, 0.07850666, 0.08339743, 0.08121902],
        [0.11200925, 0.12436181, 0.25201148, 0.15984112, 0.07940041, 0.11406696, 0.07786476, 0.08044427],
        [0.10719867, 0.10662822, 0.17096704, 0.26908734, 0.08279348, 0.11584437, 0.07854924, 0.06893167],
        [0.08888885, 0.08260953, 0.11154103, 0.10980257, 0.37566236, 0.08878891, 0.0773462, 0.06536062],
        [0.09239034, 0.08750129, 0.14101669, 0.13506968, 0.07813732, 0.3162152, 0.07107523, 0.07859427],
        [0.08793714, 0.09217284, 0.09846857, 0.09304029, 0.07440785, 0.0787989, 0.39110672, 0.08406771],
        [0.08924655, 0.09763554, 0.10722301, 0.08361196, 0.06377, 0.08662114, 0.09213529, 0.3797565]
    ])
    print(weights_np)

    # 3. 绘制热力图
    plt.figure(figsize=(7.5, 6))
    sns.heatmap(weights_np, cmap="Reds", annot=False, fmt=".2f", linewidths=0.5, vmin=0, vmax=0.5)

    new_list_datasets = ['Tox21', 'Toxcast', 'SIDER', 'ClinTox', 'BBBP', 'BACE', 'HIV', 'MUV']
    y_labels = new_list_datasets
    x_labels = new_list_datasets

    plt.xticks(ticks=np.arange(len(x_labels)) + 0.5, labels=x_labels, rotation=45, fontsize=19)
    plt.yticks(ticks=np.arange(len(y_labels)) + 0.5, labels=y_labels, rotation=0, fontsize=19)

    # 4. 添加标题
    # plt.title("MoE Weights Heatmap", fontsize=15)
    # plt.xlabel("Task-specific expert adapters in MoE", fontsize=15)
    plt.xlabel("experts:                                                                   ", fontsize=19, labelpad=-55)
    # plt.ylabel("Name of the testing task", fontsize=15)

    # 5. 保存图片
    plt.savefig(f"heatmap_{args.gnn_type}_{args.pretrain_strategy}_515_main.pdf", dpi=300, bbox_inches='tight', pad_inches=0)  # 保存为 PNG 格式，300 DPI

    # 6. 关闭图像以释放内存
    plt.close()






    # all_acc_finetune = []
    # with open(f'./shell/{args.gnn_type}_{args.pretrain_strategy}/finetune_model.txt', 'r') as file:
    #     for line in file:
    #         match = re.search(r'Test ROC AUC score: ([\d\.]+)', line)
    #         if match:
    #             all_acc_finetune.append(float(match.group(1)))
    # Nscore = 0
    # for i in range(8):
    #     Nscore += all_acc[i] / all_acc_finetune[i]
    # Nscore = Nscore / 8 * 100
    # # print(f'Nscore: {Nscore:.2f}, file name: {file_name}')
    # print(f'Nscore: {Nscore:.2f}')


if __name__ == "__main__":
    args = load_args()
    
    
    main(args)