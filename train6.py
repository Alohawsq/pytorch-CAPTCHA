from authcode.main5_1 import *
import torchvision
net = CNN_Network()
net.to(DEVICE)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.MultiLabelSoftMarginLoss()  # 多分类损失函数
log_dir ="model6_1.pth"
labels = number + ALPHABET + alphabet
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
def train(net, device, train_loader, optimizer, epoch):
    print(epoch,"\n")
    net = net.to(device).train()
    sum_correct, n ,m= 0, 0,0
    train_loss = 0.0
    for i, (input, lable) in enumerate(train_loader):
        input, lable = input.to(device), lable.to(device)
        output = net(input)
        los = loss(output, lable)
        optimizer.zero_grad()
        los.backward()
        optimizer.step()
        train_loss += los.cpu().item()
        n += lable.shape[0]
        for data,target in train_loader:
            data = data.to(device)
            target = target.to(device)
            output = net(data)
            pre1 = torch.argmax(output[:, :62], dim=1)
            real1 = torch.argmax(target[:, :62], dim=1)
            pre2 = torch.argmax(output[:, 62:124], dim=1)
            real2 = torch.argmax(target[:, 62:124], dim=1)
            pre3 = torch.argmax(output[:, 124:186], dim=1)
            real3 = torch.argmax(target[:, 124:186], dim=1)
            pre4 = torch.argmax(output[:, 186:], dim=1)
            real4 = torch.argmax(target[:, 186:], dim=1)
            pre_label = torch.cat((pre1, pre2, pre3, pre4), 0).view(4, -1)
            real_label = torch.cat((real1, real2, real3, real4), 0).view(4, -1) #把Tensor拼接成4行x（所有realX中数据）列
            # print("pre_label",pre_label.transpose(1,0))
            # print("real_label ",real_label.transpose(1,0))
            bool_ = (pre_label == real_label).transpose(0, 1)#将数据的行和列互换位置
            # print("bool_",bool_)
            m += target.shape[0]
            for j in range(0, target.shape[0]):
                if bool_[j].int().sum().item() == 4:
                    sum_correct += 1
        #print("sum_correct",sum_correct)
        if i % 10 == 0:
            print('\nTrain [{}/{} ({:.2f}%)]\t loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
                i * len(input+1), len(train_loader.dataset),
                i * len(input) / len(train_loader.dataset) * 100.,
                train_loss, sum_correct, n,
                sum_correct / n))
        #write.add_graph(net,(input,))
        image = torchvision.utils.make_grid(input)
        write.add_image("test image", image, epoch)
        # Record training loss from each epoch into the writer
        write.add_scalar("train loss", train_loss, epoch)
        write.add_scalar("train correct", sum_correct / n, epoch)
    write.flush()

def test(net, device, test_loader):
    net = net.eval()
    sum_correct, n ,m= 0, 0,0
    test_loss = 0.0
    for i, (input, lable) in enumerate(test_loader):
        input, lable = input.to(device), lable.to(device)
        output = net(input)
        los = loss(output, lable)
        test_loss += los.cpu().item()
        n += lable.shape[0]
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = net(data)
            pre1 = torch.argmax(output[:, :62], dim=1)
            real1 = torch.argmax(target[:, :62], dim=1)
            pre2 = torch.argmax(output[:, 62:124], dim=1)
            real2 = torch.argmax(target[:, 62:124], dim=1)
            pre3 = torch.argmax(output[:, 124:186], dim=1)
            real3 = torch.argmax(target[:, 124:186], dim=1)
            pre4 = torch.argmax(output[:, 186:], dim=1)
            real4 = torch.argmax(target[:, 186:], dim=1)
            pre_label = torch.cat((pre1, pre2, pre3, pre4), 0).view(4, -1)
            real_label = torch.cat((real1, real2, real3, real4), 0).view(4, -1)  # 把Tensor拼接成4行x（所有realX中数据）列
            # print("pre_label",pre_label.transpose(1,0))
            # print("real_label ",real_label.transpose(1,0))
            bool_ = (pre_label == real_label).transpose(0, 1)  # 将数据的行和列互换位置
            # print("bool_",bool_)
            m += target.shape[0]
            for j in range(0, target.shape[0]):
                if bool_[j].int().sum().item() == 4:
                    sum_correct += 1
        if i % 10 == 0:
            print('\nTest [{}/{} ({:.4f}%)]\t loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
                i * len(input), len(train_loader.dataset),
                i * len(input) / len(train_loader.dataset) * 100.,
                test_loss, sum_correct, n,
                sum_correct / n))
        write.add_scalar("train loss", test_loss, epoch)
        write.add_scalar("train correct", sum_correct / n, epoch)
        write.flush()
        return sum_correct
if __name__ == "__main__":
    if not os.path.exists("six_2"):
        os.mkdir("six_2")
    write = SummaryWriter("six_2")
    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        start_epoch = checkpoint['epoch'] + 1
        print('Loaded epoch {} successfully!'.format(start_epoch))
    else:
        start_epoch = 0
        print('No save model, will start from scratch!')
    for epoch in range(1, epoch + 1):
        train(net, DEVICE, train_loader, optimizer, epoch)
        scheduler.step()
        state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, log_dir)
        if epoch % 10 == 0:
            for epoch in range(1,6):
                test(net, DEVICE, test_loader)
    write.close()
    # Save the Trained Model
    ckpt_dir = '../authcode'
    save_path = os.path.join(ckpt_dir, 'modelsix.pth')
    torch.save({'state_dict': net.state_dict()}, save_path)