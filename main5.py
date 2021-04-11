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
size = (90,35)
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

#width=180  90
#height=60  35

epoch = 20000000
batch_size = 100
a = 20000
lr = 0.001
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = ImageCaptcha(width=90, height=35, font_sizes=[fontsize])

class MyDataSet(Dataset):
    def __init__(self,num):
        self.num =num
        return
    def __getitem__(self, index):
        f = random_captcha()
        label = ''.join(f)
        image = generator.generate(label)
        if self.num % 2 == 0:
            for i in range(4):
                image = generator.create_noise_curve(image, getRandomColor())
            image = generator.create_noise_dots(image, getRandomColor(), 1, 14)
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

train_dataset = MyDataSet(random.randint(500,1000))
train_loader = torch.utils.data.DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
test_dataset = MyDataSet(random.randint(500,1000))
test_loader = torch.utils.data.DataLoader(test_dataset,shuffle=True,batch_size=batch_size)
#数据集准备完成

class CNN_Network(nn.Module):
    def __init__(self):
        super(CNN_Network, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, stride=1, kernel_size=3, padding=1),  # 16,90,35
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),               # 16,90,35
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, stride=1, kernel_size=3, padding=1),     # 32,90,35
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2, kernel_size=2),      # 32,45,17
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, stride=1, kernel_size=3, padding=1),    # 64,45,17
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),    # 128,45,17
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),         # 128,22,8
        )
        self.fc = nn.Sequential(
            nn.Linear(128*22*8, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 248)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


dummy_input = torch.rand(16,3,35,90)
model = CNN_Network()
with SummaryWriter(comment='Net',log_dir='five') as w:
    w.add_graph(model.eval(),(dummy_input,))

# for i, (input, lable) in enumerate(train_loader):
#     print("i", i)
#     output = model(input)
#     pre1 = torch.argmax(output[:, :62], dim=1)
#     real1 = torch.argmax(lable[:, :62], dim=1)
#     pre2 = torch.argmax(output[:, 62:124], dim=1)
#     real2 = torch.argmax(lable[:, 62:124], dim=1)
#     pre3 = torch.argmax(output[:, 124:186], dim=1)
#     real3 = torch.argmax(lable[:, 124:186], dim=1)
#     pre4 = torch.argmax(output[:, 186:], dim=1)
#     real4 = torch.argmax(lable[:, 186:], dim=1)
#     pre_lable = torch.cat((pre1, pre2, pre3, pre4), 0).view(4, -1)
#     real_label = torch.cat((real1, real2, real3, real4), 0).view(4, -1)  # 把Tensor拼接成4行x（所有realX中数据）列
#     print("pre_label", pre_lable.transpose(1, 0))
#     print("real_label", real_label.transpose(1, 0))
#     bool_ = (pre_lable == real_label).transpose(1, 0)  # 将数据的行和列互换位置
#     print("bool_:", bool_)