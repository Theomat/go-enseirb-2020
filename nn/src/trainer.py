from mcts import MCTS
from model import AlphaGoZero


class Trainer:

    def __init__(self, optimizer, replay_buffer):
        self.model = AlphaGoZero()
        self.best = AlphaGoZero()
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer

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
        inputs = [s for (s, _, _) in samples]
        outputs = [(pi, r) for (_, pi, r) in samples]
        self.optimizer.zero_grad()
        # TODO: do loss, the following doesn't work I know XD
        y_pred = self.model(inputs)

        loss.backward()
        self.optimizer.step()

        # TODO: Also add somewhere the loss or print it so we have an update

    def train(self, training_steps: int):
        for _ in range(training_steps):
            self._training_step()

    def on_checkpoint(self):
        # TODO: keep the best NN, should we really do that ? I'm a bit lazy :P
        # TODO: also save the model
        pass
