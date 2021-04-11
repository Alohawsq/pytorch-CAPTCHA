from PIL import Image
import cv2
import torch
from torchvision.transforms import transforms
from authcode.train3 import CNN_Network

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']
path = "./loadimage/khxy.png"
net=CNN_Network()
#  加载参数
ckpt = torch.load('model4.pth')
net.load_state_dict(ckpt['state_dict'])            #参数加载到指定模型cnn


def cropImage(path):
    img = cv2.imread(path, 0)
    # x, y = img.shape[0:2]
    # m = x/35
    # n = y/90
    # if (m>2) or (n>2):
    #     img = cv2.resize(img, (int(y /m), int(x /n)))
    img = cv2.resize(img,(90,35))
    print(img.shape[0:2])
    return img

img = cv2.imread(path, 0)
print(img.shape[0],img.shape[1])
if img.shape != (35, 90):
    img = cropImage(path)
    # img = cv2.imread(path, 0)
img = img / 255.
img = torch.from_numpy(img).float()
img = torch.unsqueeze(img, 0)
img = torch.unsqueeze(img, 0)
pred = net(img)
a1 = torch.argmax(pred[0, :62], dim=0)
a2 = torch.argmax(pred[0, 62:124], dim=0)
a3 = torch.argmax(pred[0, 124:186], dim=0)
a4 = torch.argmax(pred[0, 186:], dim=0)
pred = [a1, a2, a3, a4]
print(a1,"\n",a2,"\n",a3,"\n",a4)
print(pred)
labels = number + ALPHABET + alphabet
p = []
for i in pred:
    #print(labels[i.item()], end='')
    p.append((labels[i.item()]).lower())
    print(p)
print("\n")
print(p)