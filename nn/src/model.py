import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaGoCnn(nn.Module):
    def __init__(self):
        super(AlphaGoCnn, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)

        self.fc1 = nn.Linear(32 * 9 * 9, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):

        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.1)

        x = x.view(-1, 32 * 9 * 9)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.1)
        x = torch.sigmoid(self.fc3(x)).reshape(-1)

        return x


class ConvolutionalBlock(nn.Module):
    def __init__(self, features=15):
        super(ConvolutionalBlock, self).__init__()
        self.conv = nn.Conv2d(features, 256, 3, padding=1)
        self.bn = nn.BatchNorm2d(256)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):

    def __init__(self, blocks=9):
        super(ResidualBlock, self).__init__()

        self.layers = []
        for _ in range(blocks):
            self.layers.append({
                'conv1': nn.Conv2d(256, 256, 3, padding=1),
                'bn1': nn.BatchNorm2d(256),
                'conv2': nn.Conv2d(256, 256, 3, padding=1),
                'bn2': nn.BatchNorm2d(256)
            })

    def forward(self, x):

        ipt = x.clone()
        for layer in self.layers:
            x = F.relu(layer['bn1'](layer['conv1'](x)))
            x = F.relu(layer['bn2'](layer['conv2'](x)))
            x.add_(ipt)
            x = F.relu(x)

        return x


class PolicyNN(nn.Module):

    def __init__(self):
        super(PolicyNN, self).__init__()

        self.conv = nn.Conv2d(256, 2, 1)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * 9 * 9, 9 * 9 + 1)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, 2 * 9 * 9)
        return self.fc(x)


class ValueNN(nn.Module):

    def __init__(self):
        super(ValueNN, self).__init__()

        self.conv = nn.Conv2d(256, 2, 1)
        self.bn = nn.BatchNorm2d(2)
        self.fc1 = nn.Linear(2 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, 2 * 9 * 9)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))


class AlphaGoZero(nn.Module):
    def __init__(self, features=15, residual=9):
        super(AlphaGoZero, self).__init__()

        self.convolutional_block = ConvolutionalBlock(features=features)
        self.residual_block = ResidualBlock(blocks=9)
        self.policy_head = PolicyNN()
        self.value_head = ValueNN()

    def forward(self, x):
        x = self.convolutional_block(x)
        x = self.residual_block(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v
