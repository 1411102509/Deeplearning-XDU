import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms #transform模块提供了一般的图像转换操作类

import matplotlib.pyplot as plt
import numpy as np
import time
import os

#**************************************** 处理数据 *********************************************
#数据增强，为提高模型的泛化能力，一般会将数据进行增强操作
data_dir = "..\data"
transform_test = transforms.Compose([transforms.ToTensor()])
transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  #左右翻转
        transforms.RandomGrayscale(),       #灰度随机变换
        transforms.ToTensor()               #转换成tensor类型
    ]
)

#下载数据集
train_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, 
                    transform=transform_train, download=True)
test_dataset  = torchvision.datasets.FashionMNIST(root=data_dir, train=False,
                    transform=transform_test,  download=True)

#加载数据集
train_dataloader = DataLoader(dataset=train_dataset, batch_size=100, 
                    shuffle=True,  num_workers=4)    #shuffle洗牌，num_workers:进程数
test_dataloader  = DataLoader(dataset=test_dataset,  batch_size=50, 
                    shuffle=False, num_workers=4)


'''# 随机显示一个训练集batch
print("train_dataset_Size:",train_dataset.data.size())
print("train_label_Size:",train_dataset.targets.size())
plt.imshow(train_dataset.data[0].numpy(),cmap = 'gray')
plt.title('%i'%train_dataset.targets[0])
plt.show()

# 随机显示一个训练集batch
print("test_dataset_Size:",test_dataset.data.size())
print("test_label_Size:",test_dataset.targets.size())
plt.imshow(test_dataset.data[0].numpy(),cmap = 'gray')
plt.title('%i'%test_dataset.targets[0])
plt.show()'''


#**************************************** 搭建模型 ***********************************************
class Net(nn.Module):
    #构建网络
    def __init__(self):
        super(Net,self).__init__()  #调用父类,继承       #28*28*1
        self.conv1 = nn.Conv2d(1, 64, 1, padding = 1)   #30*30*64
        self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)  #30*30*64
        self.conv3 = nn.Conv2d(64, 64, 3, padding = 1)  #30*30*64
        self.pool1 = nn.MaxPool2d(2,2)                  #15*15*64
        self.bn1 = nn.BatchNorm2d(64)   #降低过拟合     
        self.relu1 = nn.ReLU()

        self.conv4 = nn.Conv2d(64,128,3,padding=1)      #15*15*128
        self.conv5 = nn.Conv2d(128, 128, 3,padding=1)   #15*15*128
        self.conv6 = nn.Conv2d(128, 128, 3,padding=1)   #15*15*128
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)      #8*8*128
        self.bn2 = nn.BatchNorm2d(128)  #降低过拟合  
        self.relu2 = nn.ReLU()

        self.fc5 = nn.Linear(128*8*8,512)               #512
        self.drop1 = nn.Dropout2d()     #降低过拟合
        self.fc6 = nn.Linear(512,10)                    #10

    #前向传播
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = x.view(-1,128*8*8)          #将多行Tensor拼接成一行
        x = self.fc5(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.fc6(x)

        return x

    #LeNet-5    %85
    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 6, 5)     #24*24*6
    #     self.conv2 = nn.Conv2d(6, 16, 5)    #8*8*6
    #     self.fc1   = nn.Linear(16*4*4, 120)  
    #     self.fc2   = nn.Linear(120, 84)
    #     self.fc3   = nn.Linear(84, 10)

    # def forward(self, x): 
    #     x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) 
    #     x = F.max_pool2d(F.relu(self.conv2(x)), 2) 
    #     x = x.view(x.size()[0], -1) 
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)        
    #     return x

    #AlexNet
    # def __init__(self, num_classes=10):
    #     super(Net, self).__init__()
    #     self.features = nn.Sequential(
    #         nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(kernel_size=3, stride=2),
    #         nn.Conv2d(64, 192, kernel_size=5, padding=2),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(kernel_size=3, stride=2),
    #         nn.Conv2d(192, 384, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(384, 256, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(256, 256, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(kernel_size=3, stride=2),
    #     )
    #     self.classifier = nn.Sequential(
    #         nn.Linear(256 * 6 * 6, 4096),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(4096, 4096),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(4096, num_classes),
    #     )

    # def forward(self, x):
    #     x = self.features(x)
    #     x = x.view(x.size(0), 256 * 6 * 6)
    #     x = self.classifier(x)
    #     return x

    #制定训练策略
    def train_sgd(self, device, epochs = 30):
        optimizer = optim.Adam(self.parameters(),lr=0.0001) #选择Adam作为优化器，学习率0.0001

        path = "weights.tar"
        init_epoch = 0      #训练集中的数据被训练的次数

        #加载保存的模型数据
        if os.path.exists(path) is not True:
            loss = nn.CrossEntropyLoss()    #交叉熵损失
        else:
            check_point = torch.load(path)
            self.load_state_dict(check_point['model_state_dict'])
            optimizer.load_state_dict(check_point['optimizer_state_dict'])
            init_epoch = check_point['epoch']+1
            # print(init_epoch)
            loss = check_point['loss']

        for epoch in range(init_epoch,epochs):
            timestart = time.time()

            running_loss = 0.0
            total = 0
            correct = 0
            for i, data in enumerate(train_dataloader, 0):
                #获取输入
                inputs, labels = data
                inputs, labels = inputs.to(device),labels.to(device)

                #清空导数参数
                optimizer.zero_grad()

                #前向传播，反向传播，更新优化器
                outputs = self(inputs)
                l = loss(outputs, labels)
                l.backward()
                optimizer.step()

                running_loss += l.item()

                if i % 500 == 499:  # print every 500 mini-batches
                    print('[%d, %5d] loss: %.4f' %
                            (epoch, i, running_loss / 500))
                    running_loss = 0.0
                    
                    garbage, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    print('Accuracy of the network on the %d train images: %.3f %%' % (total,
                            100.0 * correct / total))
                    total = 0
                    correct = 0
                    #保存网络的训练状态
                    torch.save({'epoch':epoch,
                                'model_state_dict':net.state_dict(),
                                'optimizer_state_dict':optimizer.state_dict(),
                                'loss':loss
                                },path)

            print('epoch %d cost %3f sec' %(epoch,time.time()-timestart))
            print('Finished Training')
    
    #测试
    def test(self,device):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_dataloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                garbage, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %.3f %%' % (
                100.0 * correct / total))

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:",device)
    net = Net()
    net = net.to(device)
    net.train_sgd(device,30)
    net.test(device)



