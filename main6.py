import cv2
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from authcode.make import *
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

#width=180  90
#height=60  35

epoch = 20000000
batch_size = 256
lr = 0.001
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyDataSet(Dataset):
    def __init__(self,num):
        all_data = []
        all_labels = []
        self.num = num
        for i in range(num):
            f = genImage2()
            label = f[0]
            img = f[1]
            img = np.array(img)
            img = img / 255.
            img = torch.from_numpy(img).float()
            label = torch.from_numpy(one_hot(label)).float()
            all_data.append(img)
            all_labels.append(label)
            i = iter(all_data)
            j = iter(all_labels)
            self.imgname = next(i)[2]
            self.imglabel = next(j)[2].view(-1)
    def __getitem__(self, index):
        if True:
            img = self.imgname[index]
            label = self.imglabel[index]
            img = torch.unsqueeze(img, 0)
            return img, label
    def __len__(self):
        return len(self.imgname)
train_dataset = MyDataSet(1000)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = MyDataSet(1000)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)
#数据集准备完成

class CNN_Network(nn.Module):
    def __init__(self):
        super(CNN_Network, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, stride=1, kernel_size=3, padding=1),  # 16,90,35
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
            nn.Linear(22528, 2048),
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

net = CNN_Network()
net.to(DEVICE)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.MultiLabelSoftMarginLoss()

dummy_input = torch.rand(16,1,35,90)
model = CNN_Network()
with SummaryWriter(comment='Net',log_dir='six') as w:
    w.add_graph(model,(dummy_input,))