from mcts import MCTS
from model import AlphaGoZero
from loss import alpha_go_zero_loss

import time

import torch
from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self, optimizer, replay_buffer):
        # autodetect best device
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Training on {'cpu' if self.device == 'cpu' else torch.cuda.get_device_name(0)}")

        self.model = AlphaGoZero().to(self.device)
        self.best = AlphaGoZero().to(self.device)
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer

        self.writer = SummaryWriter()

        # the number of training steps between each checkpoint
        self.checkpoint: int = 1000
        self._train_steps_since_last_checkpoint: int = 0
        self.sample_size: int = 2048

    def produce_episodes(self, episodes: int = 25000):
        mcts = MCTS(self.model)
        # Maybe use tqdm ?
        for _ in range(episodes):
            self.replay_buffer.store(mcts.self_play())

    def _training_step(self):
        self._train_steps_since_last_checkpoint += 1
        if self._train_steps_since_last_checkpoint == self.checkpoint:
            self.on_checkpoint()

        samples = self.replay_buffer.sample(self.sample_size)
        inputs = torch.tensor([s for (s, _, _) in samples])

        # TODO: we need a faster way to do it (with pytorch when using tensors), this is just the dumb (correct) way
        y_target = [(pi, r) for (_, pi, r) in samples]
        pi, z = list(zip(*y_target))

        pi = torch.tensor(pi)
        z = torch.tensor(z)

        self.optimizer.zero_grad()

        # TODO: do loss, the following doesn't work I know XD
        p, v = self.model(inputs)

        # where p and v are the predicted values and pi and z the target values
        loss = alpha_go_zero_loss(p, v, pi, z)

        loss.backward()
        self.optimizer.step()

        # TODO: Also add the loss or print it so we have an update
        # writer.add_scalars('loss', {'train_loss': running_loss / 100, 'test_los': test_loss}, n_iter)

    def train(self, training_steps: int):
        for _ in range(training_steps):
            self._training_step()

    def on_checkpoint(self):
        # TODO: keep the best NN, should we really do that ? I'm a bit lazy :P (just keep the last one, fuck it XD)

        # Save the best model
        torch.save(self.best.state_dict(), f'alphago_zero_model_{int(time.time())}.pt')
