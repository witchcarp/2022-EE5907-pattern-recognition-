import torch.nn as nn
import torch.nn.functional as F


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.con1 = nn.Conv2d(1, 20, kernel_size=5)
#         self.pool = nn.MaxPool2d(2, stride=2)
#         self.con2 = nn.Conv2d(20, 50, kernel_size=5)
#         self.fc = nn.Linear(1250, 500)
#         self.fc1 = nn.Linear(500, 26)
#
#     def forward(self, x):
#         x = self.pool(self.con1(x))
#         x = self.pool(self.con2(x))
#         x = F.relu(self.fc(x.reshape(256, -1)))
#         x = F.relu(self.fc1(x))
#
#         return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.con1 = nn.Conv2d(1, 20, kernel_size=5)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc = nn.Linear(20*14*14, 500)
        self.fc1 = nn.Linear(500, 26)

    def forward(self, x):
        x = self.pool(self.con1(x))
        x = F.relu(self.fc(x.reshape(256, -1)))
        x = F.relu(self.fc1(x))

        return x


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.con1 = nn.Conv2d(1, 20, kernel_size=5)  # 256 20 28 28
#         self.con2 = nn.Conv2d(20, 50, kernel_size=5) # 256 50 24 24
#         self.fc = nn.Linear(50*24*24, 500)
#         self.fc1 = nn.Linear(500, 26)
#
#     def forward(self, x):
#         x = self.con1(x)
#         x = self.con2(x)
#         x = F.relu(self.fc(x.reshape(256, -1)))
#         x = F.relu(self.fc1(x))
#
#         return x