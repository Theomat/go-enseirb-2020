from trainer import Trainer
from uniform_replay_buffer import UniformReplayBuffer

import argparse
import sys


parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Train the AlphaZero model.")
parser.add_argument('--model', type=str, help="the path of the base model")
parser.add_argument('-o', '--output', help="output model file", default=None)
parser.add_argument('--epochs', type=int, default=10, help="number of epochs")
parser.add_argument('--lr', help='learning rate', default=1e-3, type=float)
parser.add_argument('--gpu', help='gpu device number', default=0, type=int)
parser.add_argument('--replay-size', help='replay buffer size', default=1e5, type=int)
parser.add_argument('--episodes', type=int, default=20, help="number of episodes per epoch")
parser.add_argument('--checkpoints', default=5, type=int, help="number of epochs between checkpoints")
parser.add_argument('--batch-size', default=1024, type=int, help="batch size")
parser.add_argument('--recursion-limit', default=2000, type=int, help="maximum recursion depth")


# parser.add_argument('-v', '--verbose', dest='verbose', action='store_const', const=True, default=False)

parameters = parser.parse_args(sys.argv[1:])
parameters.output = parameters.output or parameters.model

sys.setrecursionlimit(parameters.recursion_limit)

buffer = UniformReplayBuffer(size=parameters.replay_size)
trainer = Trainer(buffer, file=parameters.model, episodes_per_step=parameters.episodes, lr=parameters.lr,
                  cuda=parameters.gpu, batch_size=parameters.batch_size)
trainer.train(parameters.epochs)
trainer.save(parameters.output)
