# import torch

# # 创建两个需要梯度的张量
# x = torch.nn.Parameter(torch.randn(2, 2))
# y = torch.randn(2, 2)
# a = x

# # 进行一些操作
# z = a ** 2
# z = z.mean()
# z.backward()

# # 查看 z 的 grad_fn
# print(z.grad_fn)
# print(x.grad)
# print(y.grad)  
# print(a.grad)
# if isinstance(a, torch.nn.Parameter):
#     print('yes')

import torch
import torch.nn as nn
import torch.optim as optim

# **模型 B：用于计算模型 A 的参数**
class ModelB(nn.Module):
    def __init__(self):
        super(ModelB, self).__init__()
        self.fc1 = nn.Linear(5, 200)  # 输出足够的元素数
        self.fc2 = nn.Linear(5, 100)
        self.fc_bias = nn.Linear(5, 10)

    def forward(self, x):
        batch_size = x.shape[0]  # 获取 batch_size
        w1 = self.fc1(x).view(batch_size, 10, 20)  # 变换成正确的形状
        w2 = self.fc2(x).view(batch_size, 10, 10)
        b = self.fc_bias(x).view(batch_size, 10)
        return {"fc1.weight": w1, "fc2.weight": w2, "fc2.bias": b}

# **模型 A**
class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# **初始化模型**
model_b = ModelB()
model_a = ModelA()

# **输入数据**
x_input = torch.randn(4, 20)  # 进入 A 的数据
b_input = torch.randn(4, 5)  # 进入 B 的数据
y_true = torch.randn(4, 10)  # 目标输出

# **计算 A 的参数（由 B 计算）**
param_dict = model_b(b_input)

# **动态更新 A 的参数**
with torch.no_grad():
    setattr(model_a.fc1, 'weight', param_dict["fc1.weight"].mean(dim=0))
    model_a.fc1.weight.copy_(param_dict["fc1.weight"].mean(dim=0))  # 取 batch 维度均值
    model_a.fc2.weight.copy_(param_dict["fc2.weight"].mean(dim=0))
    model_a.fc2.bias.copy_(param_dict["fc2.bias"].mean(dim=0))

# **前向传播**
y_pred = model_a(x_input)

# **计算损失**
loss_fn = nn.MSELoss()
loss = loss_fn(y_pred, y_true)

# **反向传播**
loss.backward()

# **检查梯度**
print("model_b.fc1.weight.grad:", model_b.fc1.weight.grad)

# **更新模型 B 的参数**
optimizer_b = optim.Adam(model_b.parameters(), lr=0.01)
optimizer_b.step()