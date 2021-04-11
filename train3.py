import torchvision

from authcode.main2 import *

net = CNN_Network()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.MultiLabelSoftMarginLoss()  # 多分类损失函数


def train(net, train_iter, test_iter, optimizer, loss, device, num_epochs):
    net = net.to(device)
    flag = 0.0
    for epoch in range(num_epochs):
        train_loss, test_loss, n, m = 0.0, 0.0, 0, 0
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss += l.cpu().item()
            n += y.shape[0]
        for label, target in test_iter:
            label = label.to(DEVICE)
            target = target.to(DEVICE)
            output = net(label)
            tloss = loss(output,target)
            optimizer.zero_grad()
            tloss.backward()
            optimizer.step()
            test_loss += tloss.cpu().item()
            m += target.shape[0]
        print("train_loss=", train_loss)
        train_acc = get_acc(net, train_iter, device)
        test_acc = get_acc(net, test_iter, device)
        print("test_acc=", test_acc)
        if test_acc > flag:
            ckpt_dir = '../authcode'
            save_path = os.path.join(ckpt_dir, 'model3.pth')
            torch.save({'state_dict': net.state_dict()}, save_path)
        flag = test_acc
        # Record training loss from each epoch into the writer
        write.add_scalar('Train/Loss\\', train_loss, epoch)
        write.add_scalar('Train/Accuracy\\', train_acc, epoch)
        write.add_scalar('Test/Loss\\', test_loss, epoch)
        write.add_scalar('Test/Accuracy\\', test_acc, epoch)
        write.flush()
def get_acc(net, data_iter, device):
    acc_sum, n = 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            pre1 = torch.argmax(y_hat[:, :62], dim=1)
            real1 = torch.argmax(y[:, :62], dim=1)
            pre2 = torch.argmax(y_hat[:, 62:124], dim=1)
            real2 = torch.argmax(y[:, 62:124], dim=1)
            pre3 = torch.argmax(y_hat[:, 124:186], dim=1)
            real3 = torch.argmax(y[:, 124:186], dim=1)
            pre4 = torch.argmax(y_hat[:, 186:], dim=1)
            real4 = torch.argmax(y[:, 186:], dim=1)
            pre_lable = torch.cat((pre1, pre2, pre3, pre4), 0).view(4, -1)
            real_label = torch.cat((real1, real2, real3, real4), 0).view(4, -1)
            bool_ = (pre_lable == real_label).transpose(0, 1)
            n += y.shape[0]
            for i in range(0, y.shape[0]):
                if bool_[i].int().sum().item() == 4:
                    acc_sum += 1
        return acc_sum / n


if __name__ == "__main__":
    if not os.path.exists("log"):
        os.mkdir("log")
    write = SummaryWriter("log")
    train(net, train_loader, test_loader, optimizer, loss, DEVICE, epoch)
    write.close()