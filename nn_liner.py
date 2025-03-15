import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset=torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader=DataLoader(dataset,batch_size=64)

class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.linear1=Linear(196608,10)

    def forward(self,input):
        output=self.linear1(input)
        return output

mymodel=Mymodel()

for data in dataloader:
    imgs,targets=data
    print(imgs.shape)
    output=torch.flatten(imgs)
    print(output.shape)
    output=mymodel(output)
    print(output.shape)