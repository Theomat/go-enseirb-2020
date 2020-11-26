import pickle
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import AlphaGoZero
from loss import alpha_go_zero_loss
from uniform_replay_buffer import UniformReplayBuffer
from tqdm import tqdm


f = open('./samples.npy', 'rb')
total_samples = pickle.load(f)
f.close()

LR = 0.001

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AlphaGoZero(residual=9).float().to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)

writer = SummaryWriter()

buffer = UniformReplayBuffer(len(total_samples))
buffer.store(total_samples)


for epoch in tqdm(range(1000)):

    samples = buffer.sample(128)

    inputs = torch.tensor([s for (s, _, _) in samples]).float().to(device)

    y_target = [(pi, r) for (_, pi, r) in samples]
    pi, z = list(zip(*y_target))

    pi = torch.tensor(pi).float().to(device)
    z = torch.tensor(z).float().to(device)

    optimizer.zero_grad()

    p, v = model(inputs)

    loss = alpha_go_zero_loss(p, v, pi, z)
    loss.backward()

    optimizer.step()

    running_loss = loss.item()

    writer.add_scalar('train_loss', running_loss, epoch)

torch.save(model.state_dict(), 'model.pt')
