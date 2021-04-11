import cv2
import os
import random
import numpy as np
from PIL import Image
from captcha.image import ImageCaptcha
from torch import nn
from torch.utils.data import Dataset, DataLoader
import string
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchvision import transforms
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']
#生成随机颜色背景图片
def getRandomColor():
    c1 = random.randint(0,250)
    c2 = random.randint(0, 250)
    c3 = random.randint(0, 250)
    return (c1,c2,c3)
#字体颜色，默认为蓝色
fontcolor =getRandomColor()
fontsize = 34

def one_hot(text):
    vector = np.zeros(4 * 62)  # (10+26+26)*4

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * 62 + char2pos(c)
        # print(text,i,char2pos(c),idx)
        vector[idx] = 1.0
    return vector

#定义随机方法
def random_captcha():
    #做一个容器
    captcha_text = []
    for i in range(4):
        #定义验证码字符
        c = random.choice(string.digits + string.ascii_letters)
        captcha_text.append(c)
    #返回一个随机生成的字符串
    return ''.join(captcha_text)

epoch = 200
batch_size = 500
a = 30000
lr = 0.0001
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = ImageCaptcha(width=90, height=35, font_sizes=[fontsize])

class MyDataSet(Dataset):
    def __init__(self):
        return
    def __getitem__(self, index):
        f = random_captcha()
        label = ''.join(f)
        image = generator.generate(label)
        #global img
        img = Image.open(image)
        img = np.array(img)
        #img[img >= 255] = 0
        img = img / 255.
        img = torch.from_numpy(img).float()
        #img = torch.unsqueeze(img, 0)
        img = img.permute(2, 0, 1)
        label = torch.from_numpy(one_hot(label)).float()
        return img,label
    def __len__(self):
        return a

train_dataset = MyDataSet()
train_loader = torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
test_dataset = MyDataSet()
test_loader = torch.utils.data.DataLoader(test_dataset,shuffle=True,batch_size=batch_size)
#数据集准备完成

class VGG16(nn.Module):
        def __init__(self):
            super(VGG16, self).__init__()
            #input  input(90,35,3)
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1,padding=(1, 1)),   # output:90*35*64
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=(1, 1)), # output:90*35*128
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)  # 池化后长宽减半 output:45,17*64
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1,padding=(1, 1)),  # output:45*17*256
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=(1, 1)), # output:45*17*256
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)  # 池化后长宽减半 output:22*8*128
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1,padding=(1, 1)),  # output:22*8*512
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),  # output:22*8*512
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)  # 池化后长宽减半 output:11*4*256
            )
            self.layer4 = nn.Sequential(
                # GROUP 4
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,padding=1),  # output:5*2*512
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),  # output:5*2*512
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)  # 池化后长宽减半 output:5*2*512
            )
            # self.layer5 = nn.Sequential(
            #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,padding=1),  # output:15*12*512
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1),  # output:14*14*512
            #     nn.ReLU(inplace=True),
            #     nn.MaxPool2d(2)  # 池化后长宽减半 output:7*6*512
            # )
            self.fc = nn.Sequential(
                nn.Linear(in_features=5*2*512, out_features=2048),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=2048, out_features=1024),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=1024, out_features=248)
            )
        # 定义前向传播
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            # x = self.layer5(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

dummy_input = torch.rand(64,3,35,90)
model = VGG16()
with SummaryWriter(comment='Net',log_dir='VGG_NET') as w:
    w.add_graph(model.eval(),(dummy_input,))


