import torch
import torch.nn.functional as F

input=torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]])

kernal=torch.tensor([[1,2,1],
                     [0,1,0],
                     [2,1,0]])

input=torch.reshape(input,(1,1,5,5))
kernal=torch.reshape(kernal,(1,1,3,3))

print(input.shape)
print(kernal.shape)

# 参数:
# input – input tensor of shape (minibatch,in_channels,iH,iW) 输入，tensor类型数字图像
# weight – filters of shape (out_channels, groups,in_channels/groups,kH,kW) 权重(卷积核)
# bias – optional bias tensor of shape (out_channels). Default: None
# stride – the stride of the convolving kernel. Can be a single number or a tuple (sH, sW). Default: 1 步长
# padding –implicit paddings on both sides of the input.  边缘填充
output=F.conv2d(input,kernal,stride=1)
print(output)

output2=F.conv2d(input,kernal,stride=2)
print(output2)

output3=F.conv2d(input,kernal,stride=1,padding=1)
print(output3)