import torch
from model import AlphaGoZero
from mcts import MCTS

import logging


logging.basicConfig(level=5)
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Training on {'cpu' if device == 'cpu' else torch.cuda.get_device_name(0)}")


model = AlphaGoZero().to(device)
mcts = MCTS(model)

print(mcts.self_play())
logging.shutdown()
