import pickle
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

from model import AlphaGoZero
from loss import alpha_go_zero_loss
from uniform_replay_buffer import UniformReplayBuffer

import random


f = open('./samples_aug_half.npy', 'rb')
total_samples = pickle.load(f)
f.close()


TEST_SPLIT = 500
FREQ_TEST = 10
LR = 0.001

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AlphaGoZero(residual=9).float().to(device)

random.shuffle(total_samples)

Xtrain, Xtest = train_test_split(total_samples, test_size=int(0.04 * len(total_samples)), random_state=42)


device_name = "cpu" if device == "cpu" else torch.cuda.get_device_name(0)

print('Training on' + device_name)
print(len(Xtrain))
print(len(Xtest))

# model.load_state_dict(torch.load("win_as_white.pt"))

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000)
writer = SummaryWriter()

buffer = UniformReplayBuffer(len(Xtrain))
buffer.store(np.expand_dims(Xtrain, axis=0))

test_buffer = UniformReplayBuffer(len(Xtest))
test_buffer.store(np.expand_dims(Xtest, axis=0))


for epoch in tqdm(range(7000)):

    inputs, pi, z = buffer.sample(128)

    inputs = inputs.float().to(device)
    pi = pi.float().to(device)
    z = z.float().to(device)

    optimizer.zero_grad()

    p, v = model(inputs)

    loss = alpha_go_zero_loss(p, v, pi, z)
    loss.backward()

    optimizer.step()
    # scheduler.step()

    running_loss = loss.item()

    writer.add_scalar('train_loss', running_loss, epoch)
    if epoch % FREQ_TEST == 0:

        inputs, pi, z = test_buffer.sample(256)

        inputs = inputs.float().to(device)
        pi = pi.float().to(device)
        z = z.float().to(device)

        writer.add_scalar('train_loss', running_loss, epoch)
        p, v = model(inputs)
        loss = alpha_go_zero_loss(p, v, pi, z)

        running_loss = loss.item()
        writer.add_scalar('test_loss', running_loss, epoch)


torch.save(model.state_dict(), 'model.pt')
