import os
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import time
import sys
#sys.path.append('/remote-home/yepeng2')
from task_vectors import TaskVector
#from eval import eval_single_dataset
#from args import parse_arguments

def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def apply_vector(vector, pretrained_checkpoint):#, scaling_coef=1.0):
    """Apply a task vector to a pretrained model."""
    with torch.no_grad():
        pretrained_model = torch.load(pretrained_checkpoint)
        new_state_dict = {}
        pretrained_state_dict = pretrained_model.state_dict()
        for key in pretrained_state_dict:
            if key not in vector:
                print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                continue
            new_state_dict[key] = pretrained_state_dict[key] + vector[key]
    pretrained_model.load_state_dict(new_state_dict, strict=False)
    return pretrained_model


def emr_merge(task_vectors):
    sum_param = {}
    n2p = []
    for m in range(len(task_vectors)):
        n2p_temp = task_vectors[m].vector
        n2p.append(n2p_temp)
        for n in n2p_temp:
            if n not in sum_param:
                sum_param[n] = []
            sum_param[n].append(n2p_temp[n])
    sum_param = {k: torch.stack(v, 0).mean(0) for k, v in sum_param.items()}
    vector_unified = {}
    scales = torch.zeros(len(task_vectors))
    masks = {}
    for n in sum_param:
        masks[n] = []
        flag = (sum_param[n]>0) * 2 - 1
        param_max = torch.zeros_like(n2p[0][n])
        for m in range(len(task_vectors)):
            param = task_vectors[m].vector[n]
            mask = (param * flag) > 0
            masks[n].append(mask)
            param_abs = torch.abs(mask*param)
            param_max = torch.where(param_abs>param_max, param_abs, param_max)
            scales[m] += torch.mean(torch.abs(param))
            #scales[m] += torch.sqrt(torch.mean(param**2))
        vector_unified[n] =  param_max * flag
    new_scales = torch.zeros(len(task_vectors))
    for m in range(len(task_vectors)):
        for n in vector_unified:
            p = vector_unified[n] * masks[n][m]
            new_scales[m] += torch.mean(torch.abs(p))
            #new_scales[m] += torch.sqrt(torch.mean(p**2))
    rescalers = scales / new_scales

    return vector_unified, masks, rescalers

def ties_merge(task_vectors):
    sum_param = {}
    n2p = []
    for m in range(len(task_vectors)):
        n2p_temp = task_vectors[m].vector
        n2p.append(n2p_temp)
        for n in n2p_temp:
            if n not in sum_param:
                sum_param[n] = []
            sum_param[n].append(n2p_temp[n])
    sum_param = {k: torch.stack(v, 0).mean(0) for k, v in sum_param.items()}
    vector_unified = {}
    for n in sum_param:
        flag = (sum_param[n]>0) * 2 - 1
        param_mean = torch.zeros_like(n2p[0][n])
        mask_num = torch.zeros_like(n2p[0][n])
        for m in range(len(task_vectors)):
            param = task_vectors[m].vector[n]
            mask = (param * flag) > 0
            param_abs = torch.abs(mask*param)
            param_mean += param_abs
            mask_num += mask
            #scales[m] += torch.sqrt(torch.mean(param**2))
        vector_unified[n] =  param_mean / mask_num * flag

    return vector_unified


def weight_average(task_vectors, coefficients):
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            new_vector[key] = sum(coefficients[k] * task_vectors[k].vector[key] for k in range(len(task_vectors)))
    return TaskVector(vector=new_vector)

def weight_average_train(task_vectors, coefficients):
    # with torch.no_grad():
    new_vector = {}
    for key in task_vectors[0].vector:
        new_vector[key] = torch.nn.Parameter(sum(coefficients[k] * task_vectors[k].vector[key].to(coefficients.device) for k in range(len(task_vectors))))
    return TaskVector(vector=new_vector)

# def ties_merge(task_vectors):
#     with torch.no_grad():
#         new_vector = {}
#         for key in task_vectors[0].vector:
#             stacked_params = torch.stack([tv.vector[key] for tv in task_vectors], dim=0)
#             new_vector[key] = torch.median(stacked_params, dim=0)[0]
#     return TaskVector(vector=new_vector)


'''
exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD'] # SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD
model = 'ViT-B-32'
args = parse_arguments()
args.home = 'home/emr-merging' # type your home path here
args.data_location = args.home + '/data'
args.model = model
args.save = args.home + '/checkpoints/' + model
args.logs_path = args.home + '/logs/' + model
args.batch_size = 16
pretrained_checkpoint = args.home + '/checkpoints/'+model+'/zeroshot.pt'

str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, 'log_{}_emr_merging.txt'.format(str_time_))

task_vectors = [
    TaskVector(pretrained_checkpoint, args.home + '/checkpoints/'+model+'/'+dataset_name+'/finetuned.pt') for dataset_name in exam_datasets
]

# merge models
vector_unified, masks, rescalers = emr_merge(task_vectors)

accs = []
for i, dataset in enumerate(exam_datasets):
    task_vector_recon = {}
    for n in vector_unified:
        task_vector_recon[n] =  vector_unified[n] * masks[n][i] * rescalers[i]
    image_encoder = apply_vector(task_vector_recon, pretrained_checkpoint)
    metrics = eval_single_dataset(image_encoder, dataset, args)
    log.info(str(dataset) + ':' + str(metrics.get('top1')*100)+'%')
    accs.append(metrics.get('top1')*100)
log.info('Avg ACC:' + str(np.mean(accs)) + '%')
'''