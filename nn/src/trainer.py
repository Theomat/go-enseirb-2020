from mcts import MCTS
from model import AlphaGoZero
from loss import alpha_go_zero_loss

import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange


class Trainer:

    def __init__(self, replay_buffer, lr: float = 1e-3, file: str = None,
                 episodes_per_step: int = 100, checkpoint: int = 1000, cuda: int = 0,
                 batch_size: int = 2048):
        # autodetect best device
        self.device = f"cuda:{cuda}" if torch.cuda.is_available() else "cpu"
        print(f"Training on {'cpu' if self.device == 'cpu' else 'cuda:'+ str(cuda)+ ' ' + torch.cuda.get_device_name(cuda)}")

        self.model = AlphaGoZero().to(self.device)
        # self.best = AlphaGoZero().to(self.device)
        if file:
            self.model.load_state_dict(torch.load(file, map_location=torch.device('cpu')))
            # self.best.load_state_dict(torch.load(file))

        self.replay_buffer = replay_buffer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        self.writer = SummaryWriter()

        # the number of training steps between each checkpoint
        self.checkpoint: int = checkpoint
        self._train_steps_since_last_checkpoint: int = 0
        self.sample_size: int = batch_size

        self.iter: int = 0

        self.episodes = episodes_per_step

    def produce_episodes(self, episodes: int = 25000):
        mcts = MCTS(self.model, self.device)
        for _ in trange(episodes, desc="episode"):
            self.replay_buffer.store([mcts.self_play()])

    def _training_step(self):
        self._train_steps_since_last_checkpoint += 1
        if self._train_steps_since_last_checkpoint == self.checkpoint:
            self.on_checkpoint()

        inputs, pi, z = self.replay_buffer.sample(self.sample_size)

        self.optimizer.zero_grad()

        # TODO: do loss, the following doesn't work I know XD
        p, v = self.model(inputs.float().to(self.device))

        # where p and v are the predicted values and pi and z the target values
        loss = alpha_go_zero_loss(p, v, pi.to(self.device), z.to(self.device))

        loss.backward()
        self.optimizer.step()

        # TODO: Also add the loss or print it so we have an update
        running_loss = loss.item()
        self.writer.add_scalar('train_loss', running_loss, self.iter)
        self.iter += 1

    def train(self, training_steps: int):
        for _ in trange(training_steps, desc="step"):
            self.produce_episodes(self.episodes)
            self._training_step()

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def on_checkpoint(self):
        # TODO: keep the best NN, should we really do that ? I'm a bit lazy :P (just keep the last one, fuck it XD)
        # Save the best model
        self.save(f'alphago_zero_model_{int(time.time())}.pt')
