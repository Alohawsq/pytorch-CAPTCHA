from authcode.train1 import *

path="./image/test"  #要测试照片的路径
net=CNN_Network()
#  加载参数
ckpt = torch.load('model1.pth')
net.load_state_dict(ckpt['state_dict'])            #参数加载到指定模型cnn
img= cv2.imread(path,0)
img = img/255.
img = torch.from_numpy(img).float()
img = torch.unsqueeze(img,0)
img = torch.unsqueeze(img,0)
pred = net(img)
a1 = torch.argmax(pred[0,:62],dim=0)
a2 = torch.argmax(pred[0,62:124],dim=0)
a3 = torch.argmax(pred[0,124:186],dim=0)
a4 = torch.argmax(pred[0,186:],dim=0)
pred = [a1,a2,a3,a4]
labels=number+ALPHABET+alphabet
for i in pred:
    print(labels[i.item()],end='')
print("success")
