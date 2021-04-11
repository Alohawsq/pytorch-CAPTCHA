from authcode.main5 import *
import torchvision
net = CNN_Network()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.MultiLabelSoftMarginLoss()  # 多分类损失函数


def train(net, device, train_loader, optimizer, epoch):
    net = net.to(device).train()
    acc_sum, n = 0, 0
    train_loss = 0.0
    m = 0
    for batch_idx, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)
        output = net(X)
        los = loss(output, y)
        optimizer.zero_grad()
        los.backward()
        optimizer.step()
        pre1 = torch.argmax(output[:, :62], dim=1)
        real1 = torch.argmax(y[:, :62], dim=1)
        pre2 = torch.argmax(output[:, 62:124], dim=1)
        real2 = torch.argmax(y[:, 62:124], dim=1)
        pre3 = torch.argmax(output[:, 124:186], dim=1)
        real3 = torch.argmax(y[:, 124:186], dim=1)
        pre4 = torch.argmax(output[:, 186:], dim=1)
        real4 = torch.argmax(y[:, 186:], dim=1)
        pre_lable = torch.cat((pre1, pre2, pre3, pre4), 0).view(4, -1)
        real_label = torch.cat((real1, real2, real3, real4), 0).view(4, -1)
        booll = (pre_lable.equal(real_label))
        if booll == True:
            acc_sum = acc_sum+1
        n += y.shape[0]
        train_loss += los.cpu().item()
        if epoch % 500 == 0:
            print('epoch:{}\tTrain:[{}/{} ({:.2f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                epoch,batch_idx * len(X), len(train_loader.dataset),
                batch_idx * len(X)/len(train_loader.dataset)*100., train_loss,
                acc_sum , n,acc_sum/n
            ))
        train_acc = acc_sum /n
        write.add_graph(net, (X,))
        # Record training loss from each epoch into the writer
        write.add_scalar('Train/Loss\\', train_loss, epoch)
        write.add_scalar('Train/Accuracy\\', train_acc, epoch)
        write.flush()

def test(net, device,  test_loader):
    acc_sum, n = 0, 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(test_loader):
            X = X.to(device)
            y = y.to(device)
            output = net(X)
            los = loss(output, y)
            pre1 = torch.argmax(output[:, :62], dim=1)
            real1 = torch.argmax(y[:, :62], dim=1)
            pre2 = torch.argmax(output[:, 62:124], dim=1)
            real2 = torch.argmax(y[:, 62:124], dim=1)
            pre3 = torch.argmax(output[:, 124:186], dim=1)
            real3 = torch.argmax(y[:, 124:186], dim=1)
            pre4 = torch.argmax(output[:, 186:], dim=1)
            real4 = torch.argmax(y[:, 186:], dim=1)
            pre_lable = torch.cat((pre1, pre2, pre3, pre4), 0).view(4, -1)
            real_label = torch.cat((real1, real2, real3, real4), 0).view(4, -1)
            booll = (pre_lable.equal(real_label))
            if booll == True:
                acc_sum = acc_sum + 1
            n += y.shape[0]
            test_loss += los.cpu().item()  # 将一批的损失相加
            if batch_idx % 10 == 0:
                print('\nTest [{}/{} ({:.2f}%)]\t loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                    batch_idx * len(X), len(train_loader.dataset),
                    batch_idx * len(X) / len(train_loader.dataset) * 100.,
                    test_loss, acc_sum, n,
                    acc_sum/n))
            test_acc = acc_sum / n
            # Record loss and accuracy from the test run into the writer
            write.add_scalar('Test/Loss\\', test_loss, epoch)
            write.add_scalar('Test/Accuracy\\', test_acc, epoch)
            write.flush()

if __name__ == "__main__":
    if not os.path.exists("five"):
        os.mkdir("five")
    write = SummaryWriter("five")
    for epoch in range(1, epoch + 1):
        train(net, DEVICE, train_loader, optimizer, epoch)
        if epoch % 10000 == 0:
            for j in range(100):
                test(net, DEVICE, test_loader)
    write.close()
    # Save the Trained Model
    ckpt_dir = '../authcode'
    save_path = os.path.join(ckpt_dir, 'modelfive.pth')
    torch.save({'state_dict': net.state_dict()}, save_path)