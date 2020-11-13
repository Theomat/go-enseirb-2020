from dataset import get_loaders
from model import AlphaGoCnn

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import time

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Training on {'cpu' if device == 'cpu' else torch.cuda.get_device_name(0)}")

EPOCHS = 7
LR = 0.001
EVAL_EVERY = 100

trainloader, testloader = get_loaders()

net = AlphaGoCnn().to(device)
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

        if i % EVAL_EVERY == EVAL_EVERY-1:

            with torch.no_grad():
                net.eval()

                test_loss = 0.0
                for it, test_batch in enumerate(testloader):
                    tables, targets = test_batch
                    preds = net(tables)
                    test_loss += criterion(preds, targets).item()
                test_loss = test_loss/it

            net.train()

            print('[%d, %5d]training_loss: %.3f, test_loss: %.3f' % (epoch + 1, i + 1, running_loss / EVAL_EVERY, test_loss))
            writer.add_scalars('loss', {'train_loss': running_loss / 100, 'test_los': test_loss}, n_iter)
            running_loss = 0.0
            n_iter += 1

torch.save(net.state_dict(), f'alphago_model_{int(time.time())}.pt')
print('Finished Training')
