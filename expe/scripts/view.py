import torch


datasets = ["tox21", "hiv", "muv", "bace", "bbbp", "toxcast", "sider", "clintox"]
#a = []

def load_model_parameters(model_file):
    # 加载模型文件
    checkpoint = torch.load(model_file, map_location='cpu')

    # 如果模型文件中有多个部分，比如 'model_state_dict'
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint  # 直接用 checkpoint

    #print(state_dict['gnn.x_embedding1.weight'])

    # 打印出所有的参数名
    print("Model parameters:")
    for key in state_dict.keys():
        print(key,state_dict[key].shape)
        #if key == 'gnn.gnns.1.mlp.0.weight':
            #a.append(state_dict[key])
            #print(state_dict[key])
    #print(state_dict['gnn.gnns.1.mlp.0.weight'][0][0])

# 调用函数并传入模型文件路径
'''
for data in datasets:
    print(data)
    model_file = './ftmodels/supervised_contextpred/gin_{}_sd8.pt'.format(data)  # 替换为你的模型文件路径
    load_model_parameters(model_file)
'''
#print(sum(a)/len(a))
model_file = '/public24_data/cj/CcccJjjj/ftmodels/gcn_supervised_contextpred/gcn_bace_sd0.pt'
#model_file = './router_models/sc_router_0_linkpred.pth'
#model_file = './model_gin/supervised_contextpred.pth'# 替换为你的模型文件路径
load_model_parameters(model_file)
