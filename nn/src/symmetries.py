from typing import Tuple

import torch

import numpy as np

# 8 transformations are:
# - (3) for in in [1, 2, 3]
#           rot90(board, i)
# - (4) for i in [1, 2, 3, 4]:
#           fliplr(rot90(board, i))
# - (1) color-swap


def random_symmetry(tensor) -> Tuple[torch.Tensor, int]:
    """
    Take a tensor and applies a random symmetry to it.

    Parameters
    -----------
    - tensor: the tensor on which to apply the symmetry

    Return
    -----------
    a tuple (new_tensor, i) where i allows to re-use the symmetry with ```apply_symmetry```
    """
    i: int = np.random.randint(0, 8)
    return apply_symmetry(tensor, i), i


def apply_symmetry(tensor, i: int) -> torch.Tensor:
    """
    Take a tensor and applies a specific symmetry to it.

    Parameters
    -----------
    - tensor: the tensor on which to apply the symmetry
    - i: the id of the symmetry to apply

    Return
    -----------
    the new tensor
    """
    # TODO: implement
    pass
