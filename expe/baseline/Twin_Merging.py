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
from torch.utils.data import DataLoader as DataLoaderRouter
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.utils.class_weight import compute_class_weight
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



class SimpleMLP(nn.Module):
    def __init__(self, num_clients, embedding_dims, hidden_dim=1024, class_weights=None):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(embedding_dims, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, num_clients)
        self.dropout = nn.Dropout(p=0.5)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()

    def forward(self, input, labels=None):
        x = input.float()
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc4(x)

        if labels is not None:
            loss = self.criterion(x, labels)
            return loss, x
        return x


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
    parser.add_argument('--alpha', type=float, default=0.1, help="balance parameter for clustering")
    parser.add_argument('--beta', type=float, default=0.01, help="balance parameter for alignment")

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
        for i in range(args.num_layer):
            intermediate_output_t[f'gnn.gnns.{i}.mlp.2'] = intermediate_output_t1[f'gnn.gnns.{i}.mlp.2'] - intermediate_output_t2[f'gnn.surgery_mlps.{i}.2']

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
        loss = loss_reg_backbone + args.alpha * feature_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_gtot_loss += loss_reg_backbone.item()
        all_fea_loss += feature_loss.item()
    
    return all_gtot_loss / len(loader), all_fea_loss / len(loader)


def eval(args, model, pretrainmodel, routermodel, loader, task_vectors, pretrained_state_dict):
    model.eval()
    pretrainmodel.eval()
    routermodel.eval()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()  # 临时启用训练模式
    for module in pretrainmodel.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()  # 临时启用训练模式
    
    y_true, y_scores = [], []
    for batch in loader:
        batch = batch.to(args.device)

        with torch.no_grad():
            _,_ = pretrainmodel(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            weights = routermodel(pretrainmodel.pool(feature, batch.batch))
            weights = weights.squeeze(0)
            weights = F.softmax(weights, dim=0)
            # sys.exit()
            
            # weights = weights[0, :].cpu()
            weights  = weights.mean(dim=0).cpu()
            task_params = {}
            for key in pretrained_state_dict:
                if key not in task_vectors[0].vector:
                    # print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                else:
                    task_params[key] = pretrained_state_dict[key] + sum(weights[k] * task_vectors[k].vector[key] for k in range(len(task_vectors)))
                    # task_params[key] = pretrained_state_dict[key]



            model.gnn.load_state_dict(task_params, strict=False)
   
            logits,_ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)   

            scores = torch.sigmoid(logits)
            #print(scores)
            #print(torch.isnan(scores))
        y_true.append(batch.y.view(batch.id.shape[0], -1))
        y_scores.append(scores)    

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
    avg_roc = cal_roc(y_true, y_scores)
    # print(f'avg_roc: {avg_roc}')
    # sys.exit()

    return  avg_roc


def test_one_dataset(args):
    set_seed(args.seed)    


    # dataset split & data loader  supervised_
    dataset = MoleculeDataset(args.dataset_dir + "/" + args.dataset, dataset=args.dataset)
    train_dataset, valid_dataset, test_dataset = data_split(args, dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=args.num_workers)

    
    model = GNN_graphpred(args)
    pretrainmodel = GNN_graphpred(args)
    if args.gnn_type == 'gin':
        routermodel = SimpleMLP(num_clients=8, embedding_dims=600, hidden_dim=1200)
    if args.gnn_type == 'gcn':
        routermodel = SimpleMLP(num_clients=8, embedding_dims=300, hidden_dim=600)



    def hook_fn(module, input, output):
        global feature
        feature = output

    if args.gnn_type == 'gin':
        pretrainmodel.gnn.gnns[4].mlp[1].register_forward_hook(hook_fn)
    if args.gnn_type == 'gcn':
        pretrainmodel.gnn.gnns[4].register_forward_hook(hook_fn)

    ## load models
    model_file = f'{args.model_dir}/ftmodels/{args.gnn_type}_supervised_{args.pretrain_strategy}/{args.gnn_type}_{args.dataset}_sd0.pt'
    print(model_file)
    dict_m = torch.load(model_file, map_location='cpu')
    dict_para = dict_m['model_state_dict']

    exam_datasets = ['tox21', 'toxcast', 'sider', 'clintox', 'bbbp', 'bace', 'hiv', 'muv']
    if args.gnn_type == 'gin':
        pretrained_path = f'{args.model_dir}/model_gin/supervised_{args.pretrain_strategy}.pth'
    else:
        pretrained_path = f'{args.model_dir}/model_architecture/{args.gnn_type}_supervised_{args.pretrain_strategy}.pth'
    pretrained_state_dict = torch.load(pretrained_path, map_location='cpu')
    task_vectors = [
    TaskVector(pretrained_path, f'{args.model_dir}/ftmodels/{args.gnn_type}_supervised_{args.pretrain_strategy}/{args.gnn_type}_{dataset_name}_sd0.pt') for dataset_name in exam_datasets
    ]
    for task_vector in task_vectors:
        task_vector.sparsify(0.1)
    
    router_dict = torch.load(f'./router_models/{args.gnn_type}_{args.pretrain_strategy}.pth', map_location='cpu')
    if "criterion.weight" in router_dict:
        del router_dict["criterion.weight"]



    model.load_state_dict(dict_para, strict=False)
    pretrainmodel.gnn.load_state_dict(pretrained_state_dict, strict=False)
    routermodel.load_state_dict(router_dict)


    model.to(args.device)
    pretrainmodel.to(args.device)
    routermodel.to(args.device)        
    

    # ts_data = np.load(f'./router_data/{args.gnn_type}_{args.pretrain_strategy}/test_dataset.npz',allow_pickle=True)
    # test_dataset = TensorDataset(torch.tensor(ts_data['features']), torch.tensor(ts_data['labels']))
    # test_loader = DataLoaderRouter(test_dataset, batch_size=64, shuffle=False)

    # routermodel.eval()  # 设置模型为评估模式
    # val_correct_predictions = 0
    # val_total_samples = 0

    # with torch.no_grad():
    #     for inputs, labels in test_loader:
    #         inputs, labels = inputs.to(args.device), labels.to(args.device)

    #         outputs = routermodel(inputs)

    #         _, predicted = torch.max(outputs, 1)
    #         val_correct_predictions += (predicted == labels).sum().item()
    #         print(val_correct_predictions)
    #         val_total_samples += labels.size(0)

    # val_accuracy = 100 * val_correct_predictions / val_total_samples
    # print(f"Validation Accuracy: {val_accuracy:.2f}%")
    
    # sys.exit()

    
    
    



    te_acc = eval(args, model, pretrainmodel, routermodel, test_loader, task_vectors, pretrained_state_dict)
    print(f'test acc:{te_acc:.2f} ')
    return te_acc


def create_data_for_routertraining(args):
    datasets_info = {
    "tox21": 0,
    "hiv": 1,
    "muv": 2,
    "bace": 3,
    "bbbp": 4,
    "toxcast": 5,
    "sider": 6,
    "clintox": 7
    }
    list_datasets = ['tox21', 'toxcast', 'sider', 'clintox', 'bbbp', 'bace', 'hiv', 'muv']
    list_num_tasks = [12, 617, 27, 2, 1, 1, 1, 17]
    all_train_features = []
    all_train_labels = []
    all_val_features = []
    all_val_labels = []
    all_test_features = []
    all_test_labels = []
    for dataset_name, num_tasks in zip(list_datasets, list_num_tasks):
        args.dataset = dataset_name
        args.num_tasks = num_tasks

        model_file = f'{args.model_dir}/ftmodels/{args.gnn_type}_supervised_{args.pretrain_strategy}/{args.gnn_type}_{args.dataset}_sd0.pt'
        print(model_file)
        dict_m = torch.load(model_file, map_location='cpu')
        dict_para = dict_m['model_state_dict']

        # dataset split & data loader  supervised_
        dataset = MoleculeDataset(args.dataset_dir + "/" + args.dataset, dataset=args.dataset)
        train_dataset, valid_dataset, test_dataset = data_split(args, dataset)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        model = GNN_graphpred(args)
        model.load_state_dict(dict_para, strict=False)

        if args.gnn_type == 'gin':
            pretrained_path = f'{args.model_dir}/model_gin/supervised_{args.pretrain_strategy}.pth'
        else:
            pretrained_path = f'{args.model_dir}/model_architecture/{args.gnn_type}_supervised_{args.pretrain_strategy}.pth'

        pretrained_state_dict = torch.load(pretrained_path, map_location='cpu')
        model.gnn.load_state_dict(pretrained_state_dict, strict=False)


        def hook_fn(module, input, output):
            global feature
            feature = output

            # global count
            # if count == 0:
            #     print("Layer Output:", output)
            #     print(feature.shape)
            #     count += 1






        
        if args.gnn_type == 'gin':
            model.gnn.gnns[4].mlp[1].register_forward_hook(hook_fn)
        if args.gnn_type == 'gcn':
            model.gnn.gnns[4].register_forward_hook(hook_fn)
        # model.gnn.register_forward_hook(hook_fn)
        #model.gnn.x_embedding1.register_forward_hook(hook_fn_emb_1)
        #model.gnn.x_embedding2.register_forward_hook(hook_fn_emb_2)

        model.to(args.device)

        model.eval()

        features = []
        for batch in train_loader:
            batch = batch.to(args.device)

            with torch.no_grad():
                logits,_ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                features.append(model.pool(feature, batch.batch))      
        train_features = torch.cat(features, dim=0)
        train_labels = torch.tensor([datasets_info[args.dataset]] * len(train_features))
        all_train_features.append(train_features)
        all_train_labels.append(train_labels)

        features = []
        for batch in val_loader:
            batch = batch.to(args.device)

            with torch.no_grad():
                logits,_ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                features.append(model.pool(feature, batch.batch))      
        val_features = torch.cat(features, dim=0)
        val_labels = torch.tensor([datasets_info[args.dataset]] * len(val_features))
        all_val_features.append(val_features)
        all_val_labels.append(val_labels)

        features = []
        for batch in test_loader:
            batch = batch.to(args.device)

            with torch.no_grad():
                logits,_ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                features.append(model.pool(feature, batch.batch))      
        test_features = torch.cat(features, dim=0)
        test_labels = torch.tensor([datasets_info[args.dataset]] * len(test_features))
        all_test_features.append(test_features)
        all_test_labels.append(test_labels)

    all_train_features = torch.cat(all_train_features, dim=0)
    all_train_labels = torch.cat(all_train_labels, dim=0)
    all_val_features = torch.cat(all_val_features, dim=0)
    all_val_labels = torch.cat(all_val_labels, dim=0)
    all_test_features = torch.cat(all_test_features, dim=0)
    all_test_labels = torch.cat(all_test_labels, dim=0)
    # print(all_train_features.shape)
    # print(all_train_labels.shape)
    # print(all_test_labels.shape)
    # sys.exit()




    os.makedirs(f'./router_data/{args.gnn_type}_{args.pretrain_strategy}', exist_ok=True)

    final_dataset_train = {
        'features': all_train_features.cpu().numpy(),
        'labels': all_train_labels.cpu().numpy()
    }
    np.savez(f"./router_data/{args.gnn_type}_{args.pretrain_strategy}/train_dataset.npz", **final_dataset_train)

    final_dataset_val = {
        'features': all_val_features.cpu().numpy(),
        'labels': all_val_labels.cpu().numpy()
    }
    np.savez(f"./router_data/{args.gnn_type}_{args.pretrain_strategy}/val_dataset.npz", **final_dataset_val)

    final_dataset_tst = {
        'features': all_test_features.cpu().numpy(),
        'labels': all_test_labels.cpu().numpy()
    }
    np.savez(f"./router_data/{args.gnn_type}_{args.pretrain_strategy}/test_dataset.npz", **final_dataset_tst)



def train_router(args):

    # 配置训练参数
    num_clients = 8  
    if args.gnn_type == 'gin':
        embedding_dims = 600  
    if args.gnn_type == 'gcn':
        embedding_dims = 300
    hidden_dim = 2 * embedding_dims  
    batch_size = 64
    epochs = 70
    learning_rate = 0.005


    tr_data = np.load(f'./router_data/{args.gnn_type}_{args.pretrain_strategy}/train_dataset.npz',allow_pickle=True)
    train_dataset = TensorDataset(torch.tensor(tr_data['features']), torch.tensor(tr_data['labels']))
    train_loader = DataLoaderRouter(train_dataset, batch_size=batch_size, shuffle=True)

    vl_data = np.load(f'./router_data/{args.gnn_type}_{args.pretrain_strategy}/val_dataset.npz',allow_pickle=True)
    val_dataset = TensorDataset(torch.tensor(vl_data['features']), torch.tensor(vl_data['labels']))
    val_loader = DataLoaderRouter(val_dataset, batch_size=batch_size, shuffle=False)

    labels = tr_data['labels']
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(class_weights)

    # 初始化模型
    model = SimpleMLP(num_clients=num_clients, embedding_dims=embedding_dims, hidden_dim=hidden_dim, class_weights=class_weights)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    # 训练过程
    model.to(args.device)
    class_weights = class_weights.to(args.device)

    best_val_accuracy = 0.0  # 用于跟踪验证集上的最佳准确率
    best_epoch = 0  # 保存最佳模型的路径

    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)

            # 前向传播和计算损失
            loss, outputs = model(inputs, labels)
            # if epoch == 1:
            #     print(outputs.shape)
            #     print(labels)
            # sys.exit()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算损失和准确率
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            # if epoch == 1:
            #     print(predicted)
            #     sys.exit()
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_predictions / total_samples

        # 在验证集上进行评估
        model.eval()  # 设置模型为评估模式
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(args.device), labels.to(args.device)

                outputs = model(inputs)
                loss = model.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct_predictions += (predicted == labels).sum().item()
                val_total_samples += labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct_predictions / val_total_samples

        print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), f'./router_models/{args.gnn_type}_{args.pretrain_strategy}.pth')



    print(f'best val accuracy:{best_val_accuracy},'
          f'best epoch:{best_epoch}')



def main(args):
    # all_acc = joint_train_datasets(args)



    # create_data_for_routertraining(args)
    # train_router(args)
    list_datasets = ['tox21', 'toxcast', 'sider', 'clintox', 'bbbp', 'bace', 'hiv', 'muv']
    list_num_tasks = [12, 617, 27, 2, 1, 1, 1, 17]
    all_acc = []
    for dataset_name, num_tasks in zip(list_datasets, list_num_tasks):
        args.dataset = dataset_name
        args.num_tasks = num_tasks
        acc = test_one_dataset(args)
        all_acc.append(acc)
        with open(f'./results/{args.gnn_type}_{args.pretrain_strategy}/twin_merging.txt', 'a') as file:
            file.write(f'data name: {dataset_name}\n')
            file.write(f'Test ROC AUC score: {acc}\n')
    

if __name__ == "__main__":
    args = load_args()
    
    
    main(args)