from authcode.main1 import *

net = CNN_Network()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.MultiLabelSoftMarginLoss()    #多分类损失函数
def train(net, train_iter, test_iter, optimizer, loss,device, num_epochs):
    net = net.to(device)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
    #torch.save(net.state_dict(), "yanzhengma.pkl")   #保存模型
    # Save the Trained Model
    ckpt_dir = '../authcode'
    save_path = os.path.join(ckpt_dir, 'model1.pth')
    torch.save({'state_dict': net.state_dict()}, save_path)
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train(net, train_loader, test_loader, optimizer, loss, device, epoch)
    