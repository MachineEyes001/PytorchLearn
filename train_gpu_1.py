import torch.optim.optimizer
import  torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import *
import time

# 准备数据集
train_data=torchvision.datasets.CIFAR10(root=".dataset",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data=torchvision.datasets.CIFAR10(root=".dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)

# length长度
train_data_size=len(train_data)
test_data_size=len(test_data)
# 如果train_data_size=10,训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用Dataloader加载数据集
train_dataloader=DataLoader(train_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)

# 创建网络模型
mymodel=Mymodel()
if torch.cuda.is_available():
    mymodel=mymodel.cuda()

# 损失函数
loss_fn=nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn=loss_fn.cuda()

# 优化器
learning_rate=1e-2
optimizer=torch.optim.SGD(mymodel.parameters(),lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step=0
# 记录测试的次数
total_test_step=0
# 训练的轮数
epoch=10

# 添加Tensorboard
writer=SummaryWriter("./Logs_train")
# start_time=time.time()
for i in range(epoch):
    print("--------第{}轮训练开始--------".format(i+1))

    # 训练步骤开始
    mymodel.train()
    train_iterator = tqdm(train_dataloader) # 显示进度条
    for data in train_iterator:
        imgs,targets=data
        if torch.cuda.is_available():
            imgs=imgs.cuda()
            targets=targets.cuda()
        outputs=mymodel(imgs)
        loss=loss_fn(outputs,targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step+=1
        if total_train_step%100==0:
            # end_time=time.time()
            # print(end_time-start_time)
            # print("训练次数:{},Loss:{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    # 测试步骤开始
    mymodel.eval()
    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets=data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs=mymodel(imgs)
            loss=loss_fn(outputs,targets)
            total_test_loss+=loss.item()
            accuracy=(outputs.argmax(1)==targets).sum() #argmax(1)横向。argmax(0)纵向
            total_accuracy+=accuracy

    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step+=1

    # 保存模型
    torch.save(mymodel,"mymodel_{}.pth".format(i))
    # torch.save(mymodel.state_dict(),"mymodel_{}.pth".format(i))
    print("模型已保存")

writer.close()



