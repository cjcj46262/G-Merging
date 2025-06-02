import math
import argparse
import sys
import re
import os
import contextlib
import copy


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


class BinaryClassificationEntropy(nn.Module):
    def __init__(self):
        super(BinaryClassificationEntropy, self).__init__()

    def forward(self, logits):
        probabilities = torch.sigmoid(logits)
        entropy = -probabilities * torch.log(probabilities) - (1 - probabilities) * torch.log(1 - probabilities)
        
        return entropy

criterion1 = BinaryClassificationEntropy()

# def kl_divergence_loss(output_probs, prior_probs):
#     # print(output_probs)
#     # print(torch.log(prior_probs))
#     # print(torch.log(output_probs))
#     kl_loss = output_probs * (torch.log(output_probs) - torch.log(prior_probs))
#     return kl_loss.mean()

def set_attr(obj, names, val):
    if len(names) == 1:
        # val = torch.nn.Parameter(val)
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)





def load_args():
    parser = argparse.ArgumentParser()

# seed & device
    parser.add_argument('--device_no', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting the dataset.")
   
#dataset
    parser.add_argument('--dataset_dir', type=str, default='./data', help='directory of dataset')
    parser.add_argument('--model_dir', type=str, default='./models', help='directory of finetuned models')
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
    parser.add_argument('--pretrain_strategy', type=str, default="contextpred")
    parser.add_argument('--rank', type=int, default=30)
    parser.add_argument('--index', type=int, default=0)


# train
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10,
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
    parser.add_argument('--alpha', type=float, default=0.1, help="balance parameter for clustering")
    parser.add_argument('--beta', type=float, default=0.01, help="balance parameter for alignment")

    parser.add_argument('--num_experts', type=int, default=8)

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





def eval(args, model, loader):
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()  # 临时启用训练模式
    
    y_true, y_scores = [], []
    for batch in loader:
        batch = batch.to(args.device)

        with torch.no_grad():
            #clf_logit, z, q_origin = model(batch)
            #q, q_idx = model.assign_head(q_origin) # N x tasks x head
            #scores = torch.sum(torch.sigmoid(clf_logit) * q, dim=-1)          
            logits,_ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)   
            #rep = model.get_graph_rep(batch)
            # finetuned_features = finetune_model.pool(finetune_model.gnn(batch.x, batch.edge_index, batch.edge_attr), batch.batch)
            # features[:,150:] = finetuned_features[:,150:]
            # logits = model.model.graph_pred_linear(features)
            scores = torch.sigmoid(logits)
            #print(scores)
            #print(torch.isnan(scores))
        y_true.append(batch.y.view(batch.id.shape[0], -1))
        y_scores.append(scores)    

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
    avg_roc = cal_roc(y_true, y_scores)

    return  avg_roc





def joint_train_datasets(args):

    def del_attr(obj, names):
        if len(names) == 1:
            delattr(obj, names[0])
        else:
            del_attr(getattr(obj, names[0]), names[1:])

    def set_attr(obj, names, val):
        if len(names) == 1:
            setattr(obj, names[0], val)
        else:
            set_attr(getattr(obj, names[0]), names[1:], val)

    def make_functional(mod):
        orig_params = tuple(mod.parameters())
        names = []
        for name, p in list(mod.named_parameters()):
            del_attr(mod, name.split("."))
            names.append(name)
        return orig_params, names

    def load_weights(mod, names, params):
        for name, p in zip(names, params):
            set_attr(mod, name.split("."), p)


    class AdaMerging(torch.nn.Module):
        def __init__(self, paramslist, model, names):
            super(AdaMerging, self).__init__()
            self.paramslist = paramslist
            self.model = model.gnn
            self.pool = model.pool
            self.names = names
            self.pretrain_lambdas = torch.ones(1, 1)
            prior = 0.125
            rlambdas = torch.ones(1, len(paramslist)-1) * prior  # (1 * tasks)
            self.lambdas_raw = torch.nn.Parameter(rlambdas)

            # self.classifier = []
            # for dataset_name in exam_datasets:
            #     classification_head = get_classification_head(args, dataset_name)
            #     layer_name = 'classifier_{}'.format(dataset_name)
            #     self.add_module(layer_name, classification_head.to(args.device))
            #     self.classifier.append(layer_name)

        def lambdas(self):
            task_lambdas = torch.clamp(self.lambdas_raw, min=0.0, max=1.0)
            lambdass = torch.cat((self.pretrain_lambdas, task_lambdas), 1)
            return lambdass

        def collect_trainable_params(self):
            return [self.lambdas_raw]

        # def get_classification_head(self, dataset_name):
        #     layer_name = 'classifier_{}'.format(dataset_name)
        #     classification_head = getattr(self, layer_name)
        #     return classification_head

        # def get_image_encoder(self):
        #     alph = self.lambdas()
        #     params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        #     params = tuple(p.cuda(0) for p in params)
        #     load_weights(self.model, self.names, params)
        #     return self.model

        def forward(self, inp):
            alph = self.lambdas()
            # params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[0].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
            params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[0].cpu()))) for j, p in enumerate(zip(*self.paramslist)))

            params = tuple(p.cuda(0) for p in params)         #   !!!!!!!!!!!!!!!!!!!!!
            load_weights(self.model, self.names, params)
            feature = self.model(inp)
            out = self.pool(feature, inp.batch)

            return out

    # pretrained_model = torch.load(pretrained_checkpoint)
    # pretrained_model_dic = pretrained_model.state_dict()

    # model = ModelWrapper(pretrained_model, exam_datasets)
    # model = model.to(args.device)
    # _, names = make_functional(model)

    # paramslist = []
    # paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in pretrained_model_dic.items())] # pretrain
    # paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in tv.vector.items())  for i, tv in enumerate(task_vectors)] # task vectors
    # torch.cuda.empty_cache()
    # adamerging_mtl_model = AdaMerging(paramslist, model, names, exam_datasets)

    # print('init lambda:')
    # print(adamerging_mtl_model.lambdas())
    # print('collect_trainable_params:')
    # print(list(adamerging_mtl_model.collect_trainable_params()))

    # epochs = 500
    # optimizer = torch.optim.Adam(adamerging_mtl_model.collect_trainable_params(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.)






    set_seed(args.seed)    
    list_datasets = ['tox21', 'toxcast', 'sider', 'clintox', 'bbbp', 'bace', 'hiv', 'muv']
    list_num_tasks = [12, 617, 27, 2, 1, 1, 1, 17]

    list_head = {}
    for dataset_name, num_tasks in zip(list_datasets, list_num_tasks):
        args.dataset = dataset_name
        args.num_tasks = num_tasks
        model_file = f'{args.model_dir}/ftmodels/{args.gnn_type}_supervised_{args.pretrain_strategy}/{args.gnn_type}_{args.dataset}_sd0.pt'
        # print(model_file)
        dict_m = torch.load(model_file, map_location='cpu')
        dict_para = dict_m['model_state_dict']
        model = GNN_graphpred(args)
        model.load_state_dict(dict_para, strict=False)
        list_head[dataset_name] = copy.deepcopy(model.graph_pred_linear).to(args.device)

    # adaw = ada()
    # adaw.to(args.device)

    _, names = make_functional(model.gnn)
    # print(names)
    # sys.exit()


    if args.gnn_type == 'gin':
        pretrained_path = f'{args.model_dir}/model_gin/supervised_{args.pretrain_strategy}.pth'
    else:
        pretrained_path = f'{args.model_dir}/model_architecture/{args.gnn_type}_supervised_{args.pretrain_strategy}.pth'
    pretrained_state_dict = torch.load(pretrained_path, map_location='cpu')
    pretrained_state_dict = dict(pretrained_state_dict)
    pretrained_param_dict = {}
    for key in pretrained_state_dict:
        if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
            continue
        if key in ['batch_norms.0.running_mean', 'batch_norms.0.running_var',
                    'batch_norms.1.running_mean', 'batch_norms.1.running_var', 
                    'batch_norms.2.running_mean', 'batch_norms.2.running_var', 
                    'batch_norms.3.running_mean', 'batch_norms.3.running_var', 
                    'batch_norms.4.running_mean', 'batch_norms.4.running_var']:
            continue
        pretrained_param_dict[key] = pretrained_state_dict[key]
    
    task_vectors = [
    TaskVector(pretrained_path, f'{args.model_dir}/ftmodels/{args.gnn_type}_supervised_{args.pretrain_strategy}/{args.gnn_type}_{dataset_name}_sd0.pt') for dataset_name in list_datasets
    ]
    paramslist = []
    paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in pretrained_param_dict.items())] # pretrain
    paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in tv.vector.items())  for i, tv in enumerate(task_vectors)]

    model.to(args.device)
    adamerging_mtl_model = AdaMerging(paramslist, model, names)
    print(adamerging_mtl_model.lambdas_raw.grad)
    print(adamerging_mtl_model.lambdas_raw)


    # optimizer = torch.optim.Adam(
    #     [
    #         {"params": adamerging_mtl_model.collect_trainable_params(), "lr": args.lr_ada},
    #     ],
    #     betas=(0.9, 0.999),
    #     weight_decay=0.
    # )
    # optimizer = torch.optim.Adam(alpha_model.collect_trainable_params(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.)
    optimizer = torch.optim.Adam(adamerging_mtl_model.collect_trainable_params(), lr=1e-7, betas=(0.9, 0.999), weight_decay=0.)
    loss_func = torch.nn.L1Loss()
    # loss_func = torch.nn.MSELoss()

    '''
    for dataset_name, num_tasks in zip(list_datasets, list_num_tasks):
        args.dataset = dataset_name
        args.num_tasks = num_tasks
        # dataset split & data loader  supervised_
        dataset = MoleculeDataset(args.dataset_dir + "/" + args.dataset, dataset=args.dataset)
        train_dataset, valid_dataset, test_dataset = data_split(args, dataset)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = GNN_graphpred(args)
        # print(adaw.adaweight)
        # sys.exit()
        merged_task_vector = weight_average_train(task_vectors, adaw.adaweight)
        task_params = {}
        for key in pretrained_state_dict:
            if key not in merged_task_vector.vector:
                print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                continue
            else:
                task_params[key] = pretrained_state_dict[key].to(args.device) + merged_task_vector.vector[key]
        model.gnn.load_state_dict(task_params, strict=False)

        model.graph_pred_linear.load_state_dict(list_head[dataset_name])

        model.to(args.device)

        te_acc = eval(args, model, test_loader)
        print(f'test acc:{te_acc:.2f}  dataset: {dataset_name}')
    '''

    all_acc = []
    for dataset_name, num_tasks in zip(list_datasets, list_num_tasks):
        args.dataset = dataset_name
        args.num_tasks = num_tasks
        # dataset split & data loader  supervised_
        dataset = MoleculeDataset(args.dataset_dir + "/" + args.dataset, dataset=args.dataset)
        train_dataset, valid_dataset, test_dataset = data_split(args, dataset)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


        y_true, y_scores = [], []
        for batch in test_loader:
            batch = batch.to(args.device)
            with torch.no_grad():
                feature = adamerging_mtl_model(batch) 
                # print(feature) 
                logits = list_head[dataset_name](feature)
                scores = torch.sigmoid(logits)

            y_true.append(batch.y.view(batch.id.shape[0], -1))
            y_scores.append(scores)    

        y_true = torch.cat(y_true, dim=0).cpu().numpy()
        y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
        avg_roc = cal_roc(y_true, y_scores)
        te_acc = avg_roc
        
        all_acc.append(te_acc)
        print(f'test acc:{te_acc:.2f}  dataset: {dataset_name}')


    for epoch in range(args.epochs):
        # losses = []
        # merged_task_vector = weight_average_train(task_vectors, adaw.adaweight)
        # task_params = {}
        # for key in pretrained_state_dict:
        #     if key not in merged_task_vector.vector:
        #         continue
        #     else:
        #         task_params[key] = pretrained_state_dict[key].to(args.device) + merged_task_vector.vector[key]

        for dataset_name, num_tasks in zip(list_datasets, list_num_tasks):
            args.dataset = dataset_name
            args.num_tasks = num_tasks
            # dataset split & data loader  supervised_
            dataset = MoleculeDataset(args.dataset_dir + "/" + args.dataset, dataset=args.dataset)
            train_dataset, valid_dataset, test_dataset = data_split(args, dataset)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            # merged_task_vector = weight_average_train(task_vectors, adaw.adaweight)
            # task_params = {}
            # for key in pretrained_state_dict:
            #     if key not in merged_task_vector.vector:
            #         continue
            #     else:
            #         task_params[key] = pretrained_state_dict[key].to(args.device) + merged_task_vector.vector[key]




            for batch in train_loader:
                batch = batch.to(args.device)

                feature = adamerging_mtl_model(batch) 
                # print(feature) 
                logits = list_head[dataset_name](feature)
                # print(logits)
                # sys.exit()
                loss = criterion1(logits)
                loss = loss.mean()
                if not torch.isnan(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    # print(adamerging_mtl_model.lambdas_raw.grad)
                    # sys.exit()
                    optimizer.step()


            print(adamerging_mtl_model.lambdas())

        # optimizer.zero_grad()
        # losses.backward()
        # optimizer.step()

        print(f'epoch {epoch} finished')

        if ((epoch+1) % 10) == 0:
            all_acc = []
            for dataset_name, num_tasks in zip(list_datasets, list_num_tasks):
                args.dataset = dataset_name
                args.num_tasks = num_tasks
                # dataset split & data loader  supervised_
                dataset = MoleculeDataset(args.dataset_dir + "/" + args.dataset, dataset=args.dataset)
                train_dataset, valid_dataset, test_dataset = data_split(args, dataset)

                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
                val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


                y_true, y_scores = [], []
                for batch in test_loader:
                    batch = batch.to(args.device)

                    with torch.no_grad():
                        feature = adamerging_mtl_model(batch) 
                        # print(feature) 
                        logits = list_head[dataset_name](feature)
                        scores = torch.sigmoid(logits)

                    y_true.append(batch.y.view(batch.id.shape[0], -1))
                    y_scores.append(scores)    

                y_true = torch.cat(y_true, dim=0).cpu().numpy()
                y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
                avg_roc = cal_roc(y_true, y_scores)
                te_acc = avg_roc
                
                all_acc.append(te_acc)
                print(f'test acc:{te_acc:.2f}  dataset: {dataset_name}')
    

    for acc in all_acc:
        with open(f'./results/{args.gnn_type}_{args.pretrain_strategy}/ada_merging.txt', 'a') as file:
            file.write(f'data name: {dataset_name}\n')
            file.write(f'Test ROC AUC score: {acc}\n')
    
    # torch.save(adamerging_mtl_model.lambdas_raw, f"./shell/contextpred/taskArith_surgeryV2_GTOT_hyper/adaweights.pth")
    # torch.save(adamerging_mtl_model.lambdas_raw, f"./shell/contextpred/taskArith_surgeryV2_GTOT_hyper/ada.pth")

    



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
        
    # }
    # torch.save(checkpoint, f"./shell/contextpred/taskArith_surgeryV2_GTOT_hyper/{args.dataset}_aligners.pth")
    # print(f'test acc:{te_acc:.2f} ')
    # # print(te_acc)
    return all_acc


def main(args):
    all_acc = joint_train_datasets(args)
    # sys.exit()

    # list_datasets = ['tox21', 'toxcast', 'sider', 'clintox', 'bbbp', 'bace', 'hiv', 'muv']
    # list_num_tasks = [12, 617, 27, 2, 1, 1, 1, 17]
    # list_batch_sizes = [512, 512, 16, 256, 512, 512, 64, 256]
    # list_alpha = [1,1,1,1,1,1,1,1]
    # list_lr_graph = [1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3]
    # list_lr_node = [1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3]

    # all_acc = []
    # for index, (dataset_name, num_tasks, batch_size, alpha, lr_graph, lr_node) in enumerate(zip(list_datasets, list_num_tasks, list_batch_sizes, list_alpha, list_lr_graph, list_lr_node)):
    #     args.index = index
    #     args.dataset = dataset_name
    #     args.num_tasks = num_tasks
    #     args.batch_size = batch_size
    #     args.alpha = alpha
    #     args.lr_graph = lr_graph
    #     args.lr_node = lr_node
    #     acc = test_one_dataset(args)
    #     all_acc.append(acc)
    #     # with open('./shell/supervised_contextpred/taskArith_surgeryV2_GTOT_hyper.txt', 'a') as file:
    #     #     file.write(f'data name: {dataset_name}\n')
    #     #     file.write(f'Test ROC AUC score: {acc}\n')
    # all_acc_finetune = []
    # with open(f'./results/{args.gnn_type}_{args.pretrain_strategy}/finetune_model.txt', 'r') as file:
    #     for line in file:
    #         match = re.search(r'Test ROC AUC score: ([\d\.]+)', line)
    #         if match:
    #             all_acc_finetune.append(float(match.group(1)))
    Nscore = 0
    for i in range(8):
        Nscore += all_acc[i]
    Nscore = Nscore / 8
    # print(f'Nscore: {Nscore:.2f}, file name: {file_name}')
    print(f'Nscore: {Nscore:.2f}')


if __name__ == "__main__":
    args = load_args()
    
    
    main(args)