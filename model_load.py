import torchvision
import torch
from model_save import *

# 方式1->保存方式1,加载模型
model=torch.load("vgg16_method1.pth")
print(model)

# 方式2->保存方式2,加载模型
# model=torch.load("vgg16_method2.pth") #只有参数
# print(model)
vgg16=torchvision.models.vgg16(weights=None)
vgg16.state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)

# 陷阱 需要from model_save import *才能成功加载
model=torch.load("mymodel_method1.pth")
print(model)