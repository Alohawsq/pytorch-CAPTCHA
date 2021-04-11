from authcode.main5 import *
import torchvision
net = CNN_Network()
net.to(DEVICE)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.MultiLabelSoftMarginLoss()  # 多分类损失函数
log_dir ="model7.pth"
test_flag = True
labels = number + ALPHABET + alphabet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


def get_acc(net, data_iter, device):
    acc_sum, n = 0, 0
    for data, target in data_iter:
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
        pre_lable = torch.cat((pre1, pre2, pre3, pre4), 0).view(4, -1)
        real_label = torch.cat((real1, real2, real3, real4), 0).view(4, -1)
        bool_ = (pre_lable == real_label).transpose(0, 1)
        n += target.shape[0]
        for i in range(0, target.shape[0]):
            if bool_[i].int().sum().item() == 4:
                acc_sum += 1
    return acc_sum / n

def train(net, train_loader, epoch):
    n=0
    train_loss= 0.0
    train_loss_min = np.Inf
    for i, (input, lable) in enumerate(train_loader):
        input,lable = input.to(device),lable.to(device)
        output = net(input)
        los = loss(output, lable)
        optimizer.zero_grad()
        los.backward()
        optimizer.step()
        train_loss += los.cpu().item()
        n += lable.shape[0]
        train_acc = get_acc(net, train_loader, device)
        print('Round {}: Training loss {:.4f}\t Training acc {:.2f}'.format(epoch, train_loss / n, train_acc))
    if train_loss <= train_loss_min:
        torch.save(net.state_dict(), log_dir)
        train_loss_min = train_loss


def test(net, test_loader):
    net = net.load_state_dict(torch.load(log_dir))
    sum_correct, n = 0, 0
    test_loss, test_loss_min = 0, 0
    for i, (input, lable) in enumerate(test_loader):
        input, lable = input.to(device), lable.to(device)
        output = net(input)
        los = loss(output, lable)
        test_loss += los.cpu().item()
        n += lable.shape[0]
        test_acc = get_acc(net, test_loader, device)
        print('Round {}: Test loss {:.4f}\t Test acc {:.2f}'.format(i + 1, test_loss / n, test_acc))

if __name__=="__main__":
    epoch=0
    while(True):
        epoch+=1
        print('Epoch {}'.format(epoch))
        if os.path.exists(log_dir):
            net.load_state_dict(torch.load(log_dir))
            print('Loaded epoch {} successfully!'.format(epoch))
        else:
            start_epoch = 0
            print('No save model, will start from scratch!')
        train(net,train_loader,epoch)
        if epoch % 100 == 0:
            test(net,train_loader)

