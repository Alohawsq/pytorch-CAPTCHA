from authcode.vggmain1 import *
import torchvision

net = VGG16()
net.to(DEVICE)
net.eval()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss_function = nn.MultiLabelSoftMarginLoss()  # 多分类损失函数
log_dir = "modelvgg4.pth"
labels = number + ALPHABET + alphabet
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

def get_acc(net, data_iter, device):
    net.eval()
    acc_sum= 0
    n = 0
    with torch.no_grad():
        for data, target in data_iter:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
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
        #return acc_sum
def train(net, train_loader,device,epoch):
    net.train()
    n=0
    train_loss= 0.0
    train_loss_min = np.Inf
    for step, (input, lable) in enumerate(train_loader):
        input,lable = input.to(device),lable.to(device)
        optimizer.zero_grad()
        output = net(input)
        loss = loss_function(output, lable)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # print train process
        # rate = (step + 1) / len(train_loader)
        # a = "*" * int(rate * 50)
        # b = "." * int((1 - rate) * 50)
        # print("\rEpoch {}:train loss: {:^3.0f}%[{}->{}]{:.3f}".format(epoch,int(rate * 100), a, b, loss), end="")
        n += lable.shape[0]
        train_acc = get_acc(net, train_loader, device)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch, train_loss /n, train_acc))
    # print('Round {}: Training loss {:.4f}\t Training acc {:.4f}%'.format(epoch, train_loss / n, train_acc*100))
    if train_loss <= train_loss_min:
        torch.save(net.state_dict(), log_dir)
        train_loss_min = train_loss
    write.add_scalar("train loss", train_loss/n, epoch)
    write.add_scalar("train correct", train_acc, epoch)
    write.flush()


# def test(net, device, test_loader):
#     net = net.eval()
#     n = 0
#     test_loss = 0.0
#     for i, (input, lable) in enumerate(test_loader):
#         input, lable = input.to(device), lable.to(device)
#         output = net(input)
#         los = loss_function(output, lable)
#         test_loss += los.cpu().item()
#         n += lable.shape[0]
#         test_acc = get_acc(net, test_loader, device)
#     print('Round {}: Test loss {:.4f}\t Test acc {:.4f}%'.format(i + 1, test_loss / n, test_acc*100))
#     write.add_scalar("train loss / n", test_loss / n, epoch)
#     write.add_scalar("train correct", test_acc, epoch)
#     write.flush()


if __name__ == "__main__":
    if not os.path.exists("VGGT"):
        os.mkdir("VGGT")
    write = SummaryWriter("VGGT")
    for epoch in range(1, epoch + 1):
        if os.path.exists(log_dir):
            net.load_state_dict(torch.load(log_dir))
            print('Loaded epoch {} successfully!'.format(epoch))
        else:
            print('No save model, will start from scratch!')
        train(net, train_loader, DEVICE, epoch)
        # if epoch % 10 == 0:
        #     for epoch in range(1, 6):
        #         test(net, DEVICE, test_loader)
    write.close()