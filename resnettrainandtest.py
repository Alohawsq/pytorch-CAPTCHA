# -*- coding: utf-8 -*-

from authcode.resnetmain1 import *
import numpy as np
import os
from tensorboardX import SummaryWriter

net = resnet18().to(DEVICE)
inchannel = net.fc.in_features
net.fc = nn.Linear(inchannel, 10)
net.to(DEVICE)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
writer = SummaryWriter(comment='base_scalar', log_dir='resscalar')
best_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    net.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = net(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        # if(batch_idx+1)%30 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (batch_idx + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
    print()


    net.eval()
    acc = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = net(data)
    #         test_loss += loss_function(output, target).item() # 将一批的损失相加
    #         pred = torch.max(output, dim=1)[1]
    #         correct += pred.eq(target.view_as(pred)).sum().item()
    #
    # test_loss /= len(test_loader.dataset)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     correct / len(test_loader.dataset)))

            loss = loss_function(output, target)
            predict_y = torch.max(output, dim=1)[1]
            acc += (predict_y == target).sum().item()
    val_accurate = acc / len(test_loader.dataset)
    if val_accurate > best_acc:
        best_acc = val_accurate
    print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
          (epoch + 1, running_loss / batch_idx, val_accurate))
    # Record training loss from each epoch into the writer
    writer.add_scalar('Train/Loss\\', running_loss, epoch)
    writer.flush()

    # Record loss and accuracy from the test run into the writer
    writer.add_scalar('Test/Loss\\', loss, epoch)
    writer.add_scalar('Test/Accuracy\\', acc, epoch)
    writer.flush()



#def restore_net():
    #net2 = Net()
    #net2.load_state_dict(torch.load('.Kaggle.pth'))
    #net2.eval()



# Save the Trained Model
ckpt_dir = '../mnistpy'
save_path = os.path.join(ckpt_dir, 'Resnet1.pth')
torch.save({'state_dict': net.state_dict()}, save_path)

print('Finished Training')