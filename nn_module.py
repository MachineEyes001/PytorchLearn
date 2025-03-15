import torch
from torch import nn


class Mymodel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        output=input+1
        return output

mymodel=Mymodel()
x=torch.tensor(1.0)
output=mymodel(x)
print(output)