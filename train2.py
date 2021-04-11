from authcode.main2 import *

net = CNN_Network()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.MultiLabelSoftMarginLoss()  # 多分类损失函数


def train(net, train_iter, test_iter, optimizer, loss, device, num_epochs):
    net = net.to(device)
    train_acc_list, test_acc_list, train_loss_list, test_loss_list = [], [], [], []
    flag = 0.0
    for epoch in range(num_epochs):
        train_loss, n = 0.0, 0
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
        print("train_loss=", train_loss)
        train_acc = get_acc(net, train_iter, device)
        test_acc = get_acc(net, test_iter, device)
        print("test_acc=", test_acc)
        if epoch >= 80:
            if test_acc > flag:
                ckpt_dir = '../authcode'
                save_path = os.path.join(ckpt_dir, 'model21.pth')
                torch.save({'state_dict': net.state_dict()}, save_path)
        flag = test_acc
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)


def get_acc(net, data_iter, device):
    acc_sum, n = 0, 0
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

    if not os.path.exists('img/train'):
        os.mkdir('img/train')
    if not os.path.exists('img/test'):
        os.mkdir('img/test')

    epoch = 200
    batch_size = 256
    lr = 0.001

    # train_dataset = MyDataSet('./img/train')
    # train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,num_workers=0)
    # test_dataset = MyDataSet('./img/test')
    # test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size,num_workers=0)

    net = CNN_Network()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    loss = nn.MultiLabelSoftMarginLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train(net, train_loader, test_loader, optimizer, loss, device, epoch)