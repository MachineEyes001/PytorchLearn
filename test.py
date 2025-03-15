import torch
from PIL import Image
import torchvision.transforms
from model import *

# 定义测试的设备
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_path="./image/plane.png"
image=Image.open(image_path)
print(image)
image=image.convert('RGB') # 确保图片为三通道

transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),torchvision.transforms.ToTensor()])
image=transform(image)
print(image.shape)

model=torch.load("mymodel_29.pth")
model.to(device) # 训练时使用GPU，验证时也需要GPU
# 或 model=torch.load("mymodel_9.pth",map_location=torch.device('cpu'))
print(model)
image=torch.reshape(image,(1,3,32,32))
image=image.to(device) # 训练时使用GPU，验证时也需要GPU
model.eval()
with torch.no_grad():
    output=model(image)
print(output)
print(output.argmax(1))