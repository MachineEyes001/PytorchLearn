import torch
import torchvision
from torch import nn
from torch.nn import ReLU,Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input=torch.tensor([[1,-0.5],
                    [-1,3]])

input=torch.reshape(input,(1,2,2))
print(input)

dataset=torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloader=DataLoader(dataset,batch_size=64)

class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.relu1=ReLU()
        self.sigmoid1=Sigmoid()

    def forward(self,input):
        output=self.sigmoid1(input)
        return output

mymodel=Mymodel()
# output=mymodel(input)
# print(output)

writer=SummaryWriter("logs_relu")

step=0
for data in dataloader:
    imgs,taregts=data
    output=mymodel(imgs)
    writer.add_images("input",imgs,global_step=step)
    writer.add_images("output",output,step)

    step+=1

writer.close()