import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear

# 搭建神经网络
class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.model1=Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64, 10),
            # nn.Softmax()
        )

    def forward(self,x):
        x=self.model1(x)
        return x

# 验证网络正确性
if __name__ == '__main__':
    mymodel=Mymodel()
    input=torch.ones((64,3,32,32))
    output=mymodel(input)
    print(output.shape)