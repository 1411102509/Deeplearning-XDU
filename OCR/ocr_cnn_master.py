# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms #transform模块提供了一般的图像转换操作类

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
import os

import cv2

class MyDataset(Dataset):               # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r', encoding='UTF-8')        # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []                       # 创建一个名为img的空列表，一会儿用来装东西
        for line in fh:
            line = line.rstrip()        # 删除本行string 字符串末尾的指定字符
            words = line.split()        # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            imgs.append((words[0], int(words[1])))  # 根据刚才txt的内容，words[0]是图片信息，words[1]是lable
            # print(int(words[1]))
        self.imgs = imgs                # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):       # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容,使得支持dataset[i]能够返回第i个数据样本这样的下标操作。
        fn, label = self.imgs[index]    # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = Image.open(fn).convert('L')
        
        # 图像预处理,主要做形态学处理，膨胀、腐蚀
        img = np.asarray(img)           #转OpenCV.Mat
        # cv2.imshow("src_img",img)

        img_id = fn.split('\\')[5][0]
        if img_id == 'b':
            _,img = cv2.threshold(img,153,255,cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            img = cv2.dilate(img,kernel,iterations=1)
            img = cv2.erode(img,kernel,iterations=2)
        elif img_id == 'g':
            _,img = cv2.threshold(img,220,255,cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            img = cv2.dilate(img,kernel,iterations=1)
            img = cv2.erode(img,kernel,iterations=2)
        else:
            _,img = cv2.threshold(img,220,255,cv2.THRESH_BINARY)
        # cv2.imshow("img",img)
        # cv2.waitKey(0)

        img = Image.fromarray(img)      #转PLT.Image
        # 预处理结束

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):                  # 这个函数也必须要写，它返回的是数据集的长度
        return len(self.imgs)




transform_test = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()])
transform_train = transforms.Compose([
        transforms.Resize((64,64)),    # 归一化大小
        transforms.RandomHorizontalFlip(),  #左右翻转
        transforms.RandomGrayscale(),       #灰度随机变换
        transforms.ToTensor()               #转换成tensor类型
    ]
)

train_dataset = MyDataset(txt_path='train.txt',transform=transform_train)
test_dataset = MyDataset(txt_path='test.txt',transform=transform_test)

train_dataloader = DataLoader(
    dataset=train_dataset, 
    batch_size=100, 
    shuffle=True,   # shuffle:洗牌
    num_workers=4   # num_workers:进程数
)       
test_dataloader = DataLoader(
    dataset=test_dataset, 
    batch_size=100, 
    shuffle=False, 
    num_workers=4
)

# print("train_dataset_Size:",len(train_dataloader))      #loader的长度是有多少个batch，所以和batch_size有关
# print("test_dataset_Size:",len(test_dataloader))

# 取几个检验
# imgs, label = train_dataset.__getitem__(0)
# print(imgs.size(),label) # batch_size, channel, height, weighttorch.Size([3, 3, 224, 224])
# imgs, label = train_dataset.__getitem__(1000)
# print(imgs.size(),label) # batch_size, channel, height, weighttorch.Size([3, 3, 224, 224])
# imgs, label = train_dataset.__getitem__(1300)
# print(imgs.size(),label) # batch_size, channel, height, weighttorch.Size([3, 3, 224, 224])




#**************************************** 搭建模型 ***********************************************

class ResidualBolock(nn.Module):
    '''
    实现子module: ResidualBlock
    '''
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=True),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut


    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

class Net(nn.Module):
    '''
    实现主module：ResNet34
    ResNet34包含多个layer，每个layer又包含多个residual block
    用子module实现residual block，用_make_layer函数实现layer
    '''
    def __init__(self, num_classes=500):
        super().__init__()
        # 前几层图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        # 重复的layer，分别由3，4，6，3个residual block
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)

        # 分类用的全连接
        self.fc = nn.Linear(512, num_classes)

    
    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        '''
        构建layer，包含多个residual block
        '''
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers = []
        layers.append(ResidualBolock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBolock(outchannel, outchannel))
        
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 2)
        x = x.view(x.size(0), -1)

        return self.fc(x)


# class Net(nn.Module):
#     #构建网络
#     def __init__(self):
#         super(Net,self).__init__()  #调用父类,继承       #28*28*1
#         self.conv1 = nn.Conv2d(1, 64, 1, padding = 1)   #30*30*64
#         self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)  #30*30*64
#         # self.drop1 = nn.Dropout2d()     #降低过拟合
#         self.conv3 = nn.Conv2d(64, 64, 3, padding = 1)  #30*30*64

#         self.pool1 = nn.MaxPool2d(2,2)                  #15*15*64
#         self.bn1 = nn.BatchNorm2d(64)   #降低过拟合     
#         self.relu1 = nn.ReLU()

#         self.conv4 = nn.Conv2d(64,128,3,padding=1)      #15*15*128
#         self.conv5 = nn.Conv2d(128, 128, 3,padding=1)   #15*15*128
#         # self.drop2 = nn.Dropout2d()     #降低过拟合
#         self.conv6 = nn.Conv2d(128, 128, 3,padding=1)   #15*15*128

#         self.pool2 = nn.MaxPool2d(2, 2, padding=1)      #8*8*128
#         self.bn2 = nn.BatchNorm2d(128)  #降低过拟合  
#         self.relu2 = nn.ReLU()

#         self.fc5 = nn.Linear(128*8*8,512)               #512
#         self.drop3 = nn.Dropout2d()     #降低过拟合
#         self.fc6 = nn.Linear(512,500)                    #500

#     #前向传播
#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         # x = self.drop1(x)
#         x = self.conv3(x)

#         x = self.pool1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)

#         x = self.conv4(x)
#         x = self.conv5(x)
#         # x = self.drop2(x)
#         x = self.conv6(x)

#         x = self.pool2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
        
#         x = x.view(-1,128*8*8)          #将多行Tensor拼接成一行
#         x = self.fc5(x)
#         x = F.relu(x)
#         x = self.drop3(x)
#         x = self.fc6(x)

#         return x

    #制定训练策略
    def train_sgd(self, device, epochs = 30):
        optimizer = optim.Adam(self.parameters(),lr=0.01) #选择Adam作为优化器，学习率0.0001

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
                # inputs = inputs.to(device)
                # labels = labels.to(device)
                #清空导数参数
                optimizer.zero_grad()

                #前向传播，反向传播，更新优化器
                outputs = self(inputs)
                l = loss(outputs, labels)
                l.backward()
                optimizer.step()

                running_loss += l.item()

                if i % 500 == 499:  # print every 500 batches
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
            self.test(device)
    
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

        print('Accuracy of the network on the test images: %.3f %%' % (
                100.0 * correct / total))

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:",device)
    net = Net()
    net = net.to(device)
    net.train_sgd(device,1)
    net.test(device)



