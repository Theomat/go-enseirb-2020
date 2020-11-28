import pickle
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from model import AlphaGoZero
from loss import alpha_go_zero_loss
from uniform_replay_buffer import UniformReplayBuffer


f = open('./samples.npy', 'rb')
total_samples = pickle.load(f)
f.close()

LR = 0.001

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AlphaGoZero(residual=9).float().to(device)

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

writer = SummaryWriter()

buffer = UniformReplayBuffer(len(total_samples))
buffer.store(np.expand_dims(total_samples, axis=0))


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

torch.save(model.state_dict(), 'model.pt')
