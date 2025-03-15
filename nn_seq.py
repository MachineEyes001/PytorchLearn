# 搭建自己的网络结构并使用Sequential管理
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()

        # # 不使用Sequential管理
        # self.conv1=Conv2d(3,32,5,padding=2)
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2=Conv2d(32,32,5,padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3=Conv2d(32,64,5,padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten=Flatten()
        # self.linear1=Linear(1024,64)
        # self.linear2 = Linear(64, 10)

        # 使用Sequential管理
        self.model1=Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64, 10)
        )

    def forward(self,x):

        # # # 不使用Sequential管理
        # x=self.conv1(x)
        # x=self.maxpool1(x)
        # x=self.conv2(x)
        # x=self.maxpool2(x)
        # x=self.conv3(x)
        # x=self.maxpool3(x)
        # x=self.flatten(x)
        # x=self.linear1(x)
        # x=self.linear2(x)

        # 使用Sequential管理
        x=self.model1(x)
        return x

mymodel=Mymodel()
print(mymodel)

# 检验网络正确性
input=torch.ones(64,3,32,32)
output=mymodel(input)
print(output.shape)

# 可视化到TensorBoard上
writer=SummaryWriter("logs_seq")
writer.add_graph(mymodel,input)
writer.close()
