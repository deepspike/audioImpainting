import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1, 24, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(24, eps=1e-4, momentum=0.9))

        self.conv2 = nn.Sequential(nn.Conv2d(24, 48, 3, stride=2, padding=1),
                                   nn.BatchNorm2d(48, eps=1e-4, momentum=0.9))

        self.conv3 = nn.Sequential(nn.Conv2d(48, 48, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(48, eps=1e-4, momentum=0.9))

        self.conv4 = nn.Sequential(nn.Conv2d(48, 96, 3, stride=2, padding=1),
                                   nn.BatchNorm2d(96, eps=1e-4, momentum=0.9))

        self.conv5 = nn.Sequential(nn.Conv2d(96, 128, 3, stride=2, padding=1),
                                   nn.BatchNorm2d(128, eps=1e-4, momentum=0.9))

        self.fc6 = nn.Sequential(nn.Linear(3072, 256),
                                 nn.BatchNorm1d(256, eps=1e-4, momentum=0.9))

        self.fc7 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 1, x.size(1), x.size(2))
        # Conv Layer
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))

        # FC Layers
        x5 = x5.view(x5.size(0), -1)
        x6 = F.relu(self.fc6(F.dropout(x5, p=0.15)))
        out = self.fc7(F.dropout(x6, p=0.15))

        return F.log_softmax(out, dim=1)
