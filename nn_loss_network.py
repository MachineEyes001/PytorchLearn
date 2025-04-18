import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

dataset=torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader=DataLoader(dataset,batch_size=1)

class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
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
        x=self.model1(x)
        return x

loss_cross=nn.CrossEntropyLoss()
mymodel=Mymodel()
for data in dataloader:
    imgs,targets=data
    outputs=mymodel(imgs)
    # print(outputs)
    # print(targets)
    result_loss=loss_cross(outputs,targets)
    # print(result_loss)
    result_loss.backward()
    print('ok')
