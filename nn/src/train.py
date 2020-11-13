from dataset import get_loaders
from model import AlphaGoCnn

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import time


EPOCHS = 10
LR = 0.001

trainloader, testloader = get_loaders()

net = AlphaGoCnn()
criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=LR)

writer = SummaryWriter()

n_iter = 0
for epoch in range(EPOCHS):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            writer.add_scalar('Loss/train', running_loss / 100, n_iter)
            # writer.add_scalar('Loss/test', np.random.random(), n_iter)
            # writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
            # writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
            running_loss = 0.0
            n_iter += 1

torch.save(net.state_dict(), f'alphago_model_{int(time.time())}.pt')
print('Finished Training')
