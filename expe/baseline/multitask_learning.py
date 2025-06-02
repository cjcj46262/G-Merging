import math
import argparse
import sys
import re
import os
import contextlib
import copy
import time

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

    # dataset
    parser.add_argument('--dataset_dir', type=str, default='./data', help='directory of dataset')
    parser.add_argument('--model_dir', type=str, default='./models', help='directory of finetuned models')
    parser.add_argument('--dataset', type=str, default='bbbp', help='root directory of dataset')
    parser.add_argument('--split', type=str, default="scaffold", help="random or scaffold or random_scaffold")

    # model
    parser.add_argument('-i', '--input_model_file', type=str, default='',
                        help='filename to read the model (if there is any)')
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
    parser.add_argument('--pretrain_strategy', type=str, default="contextpred")
    parser.add_argument('--rank', type=int, default=30)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--topk', type=int, default=8)

    # train
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')

    # optimizer
    parser.add_argument('--lr_ada', type=float, default=1e-3,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_graph', type=float, default=1e-3,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_node', type=float, default=1e-3,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')

    ## loss balance
    parser.add_argument('--alpha', type=float, default=1, help="balance parameter for two losses")
    parser.add_argument('--beta', type=float, default=0.01, help="balance parameter for alignment")
    parser.add_argument('--lamweight', type=float, default=1, help="task vectors scalar")
    parser.add_argument('--lam', action='store_true', help="Whether to use or not")

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


def train(args, model, list_ftmodel, list_train_loader, optimizer):
    # alpha_model.train()
    # finetune_model.eval()
    final_cls_loss = 0.
    # all_gtot_loss = 0.

    for ftmodel, loader in zip(list_ftmodel, list_train_loader):
        all_cls_loss = 0.
        for batch in loader:
            batch = batch.to(args.device)

            # loss = loss_reg_backbone + args.alpha * feature_loss


            _, graph_feature = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = ftmodel.graph_pred_linear(graph_feature)
            # print(pred.shape)
            # sys.exit()
            y = batch.y.view(pred.shape).to(torch.float64)
            # Whether y is non-null or not.
            is_valid = y ** 2 > 0
            loss_mat = criterion(pred.double(), (y + 1) / 2)
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            cls_loss = torch.sum(loss_mat) / torch.sum(is_valid)
            loss = cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_cls_loss += cls_loss.item()
            # all_fea_loss += feature_loss.item()
        final_cls_loss += all_cls_loss / len(loader)

    return final_cls_loss


def eval(args, model, loader):
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()  # 临时启用训练模式

    y_true, y_scores = [], []
    for batch in loader:
        batch = batch.to(args.device)

        with torch.no_grad():
            # clf_logit, z, q_origin = model(batch)
            # q, q_idx = model.assign_head(q_origin) # N x tasks x head
            # scores = torch.sum(torch.sigmoid(clf_logit) * q, dim=-1)
            logits, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            # rep = model.get_graph_rep(batch)
            # finetuned_features = finetune_model.pool(finetune_model.gnn(batch.x, batch.edge_index, batch.edge_attr), batch.batch)
            # features[:,150:] = finetuned_features[:,150:]
            # logits = model.model.graph_pred_linear(features)
            scores = torch.sigmoid(logits)
            # print(scores)
            # print(torch.isnan(scores))
        y_true.append(batch.y.view(batch.id.shape[0], -1))
        y_scores.append(scores)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
    avg_roc = cal_roc(y_true, y_scores)

    return avg_roc


def test_one_dataset(args):
    set_seed(args.seed)
    model_file = f'{args.model_dir}/ftmodels/{args.gnn_type}_supervised_{args.pretrain_strategy}/{args.gnn_type}_{args.dataset}_sd0.pt'
    print(model_file)
    dict_m = torch.load(model_file, map_location='cpu')
    dict_para = dict_m['model_state_dict']
    # print(torch.zeros(300))
    # sys.exit()
    # dict_para['gnn.batch_norms.0.running_mean'] = torch.zeros(300)
    # print(dict_para.keys())
    # sys.exit()

    # dataset split & data loader
    dataset = MoleculeDataset(args.dataset_dir + "/" + args.dataset, dataset=args.dataset)
    train_dataset, valid_dataset, test_dataset = data_split(args, dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = GNN_graphpred(args)
    # finetune_model = GNN_graphpred(args)
    model.load_state_dict(dict_para, strict=False)
    # finetune_model.load_state_dict(dict_para)



    gnn_path = f'./results/{args.gnn_type}_{args.pretrain_strategy}/multitask_learning/gnn_para.pth'
    gnn_state_dict = torch.load(gnn_path, map_location='cpu')

    model.gnn.load_state_dict(gnn_state_dict, strict=False)
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

    # list_datasets = ['tox21', 'toxcast', 'sider', 'clintox', 'bbbp', 'bace', 'hiv', 'muv']
    # for i, d_name in enumerate(list_datasets):
    #     model_file = f'./shell1/{args.gnn_type}_{args.pretrain_strategy}/taskArith_surgeryV2_GTOT/{d_name}_aligners.pth'
    #     dict_moe = torch.load(model_file, map_location='cpu')
    #     # print(dict_moe['aligner_layer0'].keys())
    #     # print(model.surgery_moe[i].state_dict().keys())
    #     # sys.exit()
    #     for layer in range(args.num_layer):
    #         model.gnn.surgery_moe_layers[layer][i].load_state_dict(dict_moe[f'aligner_layer{layer}'])
    #     model.surgery_moe[i].load_state_dict(dict_moe['aligner_graph'])
    model = model.to(args.device)

    te_acc = eval(args, model, test_loader)
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
    print(f'test acc:{te_acc:.2f} ')
    # print(te_acc)
    return te_acc


def joint_train(args):
    set_seed(args.seed)



    list_datasets = ['tox21', 'toxcast', 'sider', 'clintox', 'bbbp', 'bace', 'hiv', 'muv']
    list_num_tasks = [12, 617, 27, 2, 1, 1, 1, 17]
    list_batch_sizes = [512, 512, 16, 256, 512, 512, 64, 256]
    list_train_loader = []
    list_ftmodel = []
    for dataset_name, num_tasks, batch_size in zip(list_datasets, list_num_tasks, list_batch_sizes):
        args.dataset = dataset_name
        args.num_tasks = num_tasks
        args.batch_size = batch_size

        model_file = f'{args.model_dir}/ftmodels/{args.gnn_type}_supervised_{args.pretrain_strategy}/{args.gnn_type}_{args.dataset}_sd0.pt'
        dict_m = torch.load(model_file, map_location='cpu')
        dict_para = dict_m['model_state_dict']
        ftmodel = GNN_graphpred(args)
        ftmodel.load_state_dict(dict_para)
        ftmodel.to(args.device)
        list_ftmodel.append(ftmodel)


        # dataset split & data loader
        dataset = MoleculeDataset(args.dataset_dir + "/" + args.dataset, dataset=args.dataset)
        train_dataset, valid_dataset, test_dataset = data_split(args, dataset)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        list_train_loader.append(train_loader)

    model = GNN_graphpred(args)

    if args.gnn_type == 'gin':
        pretrained_path = f'{args.model_dir}/model_gin/supervised_{args.pretrain_strategy}.pth'
    else:
        pretrained_path = f'{args.model_dir}/model_architecture/{args.gnn_type}_supervised_{args.pretrain_strategy}.pth'


    pretrained_state_dict = torch.load(pretrained_path, map_location='cpu')

    model.gnn.load_state_dict(pretrained_state_dict, strict=False)
    model.to(args.device)

    optimizer = torch.optim.Adam(
        [
            {"params": model.gnn.parameters(), "lr": args.lr_graph},
            # {"params": model.gnn.surgery_mlps.parameters(), "lr": args.lr_node},
        ],
        betas=(0.9, 0.999),
        weight_decay=0.
    )
    # optimizer = torch.optim.Adam(alpha_model.collect_trainable_params(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.)
    loss_func = torch.nn.L1Loss()
    # loss_func = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        cls_loss = train(args, model, list_ftmodel, list_train_loader, optimizer)
        print(f'epoch {epoch} finished, cls loss: {cls_loss}')

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

    # te_acc = eval(args, model, test_loader)
    # checkpoint = {
    #     'aligner_layer0': model.gnn.surgery_mlps[0].state_dict(),
    #     'aligner_layer1': model.gnn.surgery_mlps[1].state_dict(),
    #     'aligner_layer2': model.gnn.surgery_mlps[2].state_dict(),
    #     'aligner_layer3': model.gnn.surgery_mlps[3].state_dict(),
    #     'aligner_layer4': model.gnn.surgery_mlps[4].state_dict(),
    #     'aligner_graph': model.surgery_mlp.state_dict(),
    #
    # }
    os.makedirs(f'./results/{args.gnn_type}_{args.pretrain_strategy}/multitask_learning', exist_ok=True)
    torch.save(model.gnn.state_dict(),
               f"./results/{args.gnn_type}_{args.pretrain_strategy}/multitask_learning/gnn_para.pth")
    # print(te_acc)
    acc = 0
    return acc


def main(args):
    # all_acc = joint_train_datasets(args)
    start_time = time.time()
    acc = joint_train(args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"train time : {elapsed_time:.2f} s")
    print(f"train time : {minutes:.2f} min {seconds:.2f} s")

    list_datasets = ['tox21', 'toxcast', 'sider', 'clintox', 'bbbp', 'bace', 'hiv', 'muv']
    list_num_tasks = [12, 617, 27, 2, 1, 1, 1, 17]
    list_batch_sizes = [512, 512, 16, 256, 512, 512, 64, 256]
    list_lr_graph = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]
    list_lr_node = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]


    all_acc = []
    for index, (dataset_name, num_tasks, batch_size, lr_graph, lr_node) in enumerate(
            zip(list_datasets, list_num_tasks, list_batch_sizes, list_lr_graph, list_lr_node)):
        args.index = index
        args.dataset = dataset_name
        args.num_tasks = num_tasks
        args.batch_size = batch_size
        args.lr_graph = lr_graph
        args.lr_node = lr_node
        # acc = train_one_adapters(args)
        acc = test_one_dataset(args)
        all_acc.append(acc)
        with open(f'./results/{args.gnn_type}_{args.pretrain_strategy}/multitask_learning.txt', 'a') as file:
            file.write(f'data name: {dataset_name}\n')
            file.write(f'Test ROC AUC score: {acc}\n')
    with open(f'./results/{args.gnn_type}_{args.pretrain_strategy}/multitask_learning.txt', 'a') as file:
        file.write(f"train time : {minutes:.2f} min {seconds:.2f} s\n")
        file.write(f"train time : {elapsed_time:.2f} s")


    Nscore = 0
    for i in range(8):
        Nscore += all_acc[i]
    Nscore = Nscore / 8
    # print(f'Nscore: {Nscore:.2f}, file name: {file_name}')
    print(f'Average score: {Nscore:.2f}')

    # all_acc = []
    # for index, (dataset_name, num_tasks) in enumerate(zip(list_datasets, list_num_tasks, list_batch_sizes, list_alpha, list_lr_graph, list_lr_node)):
    #     args.index = index
    #     args.dataset = dataset_name
    #     args.num_tasks = num_tasks
    #     args.batch_size = batch_size
    #     args.alpha = alpha
    #     args.lr_graph = lr_graph
    #     args.lr_node = lr_node
    #     acc = train_one_adapters(args)
    #     all_acc.append(acc)
    #     # with open('./shell/contextpred/taskArith_surgeryV2_GTOT_hyper.txt', 'a') as file:
    #     #     file.write(f'data name: {dataset_name}\n')
    #     #     file.write(f'Test ROC AUC score: {acc}\n')
    # all_acc_finetune = []
    # with open('./shell/contextpred/finetune_model.txt', 'r') as file:
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