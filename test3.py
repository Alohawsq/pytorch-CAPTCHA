import os

from PIL import Image
import cv2
import torch
from torchvision.transforms import transforms
from _train import CNN_Network

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']
path = "./loadimage/"
net=CNN_Network()
#  加载参数
#net = torch.load('model.pth')
#net.eval()
#net.load_state_dict(ckpt['state_dict'])            #参数加载到指定模型cnn
checkpoint = torch.load('model.pth')
net.load_state_dict(checkpoint['model'])
#optimizer.load_state_dict(checkpoint['optimizer'])
#epoch = checkpoint(['epoch'])


def cropImage(path):
    img = cv2.imread(path, 0)
    # x, y = img.shape[0:2]
    # m = x/35
    # n = y/90
    # if (m>2) or (n>2):
    #     img = cv2.resize(img, (int(y /m), int(x /n)))
    img = cv2.resize(img,(160,60))
    #print(img.shape[0:2])
    return img
def test(s,path):
    img = cv2.imread(path+s, 0)
    print(s)
    #print(img.shape[0],img.shape[1])
    if img.shape != (160, 60):
        img = cropImage(path+s)
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
    labels = number + ALPHABET + alphabet
    for i in pred:
        print(labels[i.item()], end='')
    print("\n")
if __name__ == "__main__":
    for s in os.listdir(path):
        test(s,path)