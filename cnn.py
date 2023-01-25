from __future__ import division
import numpy as np
from data1 import PIEDataSet1
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from net import Net
import matplotlib.pyplot as plt


if __name__ == '__main__':
    device = torch.device('cuda:0')
    net = Net()
    net.to(device)
    train = PIEDataSet1('./PIE/', if_train=True)
    trainloader = DataLoader(train, batch_size=256, shuffle=True, drop_last=True)
    test = PIEDataSet1('./PIE/', if_train=False)
    testloader = DataLoader(test, batch_size=256, shuffle=True, drop_last=True)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
    loss_train = []
    loss_test = []
    accuracy_test = []
    for epoch in range(250):
        for i, data in enumerate(trainloader, 0):
            inputs, gth = data
            optimizer.zero_grad()
            inputs = inputs.unsqueeze(1).float().to(device)
            gth = gth.float().to(device)
            outputs = net(inputs)
            loss = criterion(outputs, gth)
            loss.backward()
            optimizer.step()
        loss_train.append(loss.item())
        print("epoch:", epoch, 'train loss:', loss.item())
        correct = 0
        total = 0

        with torch.no_grad():
            for data in testloader:
                inputs, gth = data
                inputs = inputs.unsqueeze(1).float().to(device)
                gth = gth.float().to(device)
                outputs = net(inputs)
                pred = torch.max(outputs, dim=1)[1]
                groudtruth = torch.max(gth, dim=1)[1]
                h = np.count_nonzero(pred.cpu() == groudtruth.cpu())
                total = total + len(pred)
                correct = correct + h
                loss = criterion(outputs, gth)
            print('test accuracy:', h/len(pred))
            accuracy_test.append(h/len(pred))
            loss_test.append(loss.item())
            print("test loss:", (loss.item()))
    plt.subplot(1, 2, 1)
    plt.plot(np.linspace(0, 1, 250), loss_train, color='b', label='train_loss')
    plt.plot(np.linspace(0, 1, 250), loss_test, color='r', label='test_loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(np.linspace(0, 1, 250), accuracy_test, color='y', label='test_accuracy')
    plt.legend()
    plt.show()
    print('finish training')