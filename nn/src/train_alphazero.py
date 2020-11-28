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


f = open('./samples.npy', 'rb')
total_samples = pickle.load(f)
f.close()


TEST_SPLIT = .15
FREQ_TEST = 10
LR = 0.001

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AlphaGoZero(residual=9).float().to(device)

Xtrain, Xtest = train_test_split(total_samples, test_size=TEST_SPLIT, random_state=42)

# model.load_state_dict(torch.load("model_8.pt"))

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

writer = SummaryWriter()

buffer = UniformReplayBuffer(len(Xtrain))
buffer.store(np.expand_dims(Xtrain, axis=0))

test_buffer = UniformReplayBuffer(len(Xtest))
test_buffer.store(np.expand_dims(Xtest, axis=0))


for epoch in tqdm(range(1000)):

    inputs, pi, z = buffer.sample(128)

    inputs = inputs.float().to(device)
    pi = pi.float().to(device)
    z = z.float().to(device)

    optimizer.zero_grad()

    p, v = model(inputs)

    loss = alpha_go_zero_loss(p, v, pi, z)
    loss.backward()

    optimizer.step()

    running_loss = loss.item()

    writer.add_scalar('train_loss', running_loss, epoch)
    if epoch % FREQ_TEST == 0:

        inputs, pi, z = test_buffer.sample(len(Xtest))

        inputs = inputs.float().to(device)
        pi = pi.float().to(device)
        z = z.float().to(device)

        writer.add_scalar('train_loss', running_loss, epoch)
        p, v = model(inputs)
        loss = alpha_go_zero_loss(p, v, pi, z)

        running_loss = loss.item()
        writer.add_scalar('test_loss', running_loss, epoch)


torch.save(model.state_dict(), 'model_3.pt')
