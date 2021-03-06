import numpy as np
import pickle
from copy import deepcopy
from go_plot import plot_play_probabilities

ADD_SWAP = False
SWAP_VALUE = True
SWAP_CURRENT = not SWAP_VALUE

def coord_to_flat(coord):
    if coord == (-1, -1):
        return 81
    return 9 * coord[1] + coord[0]


def listify(samples):
    for i in range(len(samples)):
        samples[i] = list(samples[i])
    return samples


def tuplefy(samples):
    for i in range(len(samples)):
        samples[i] = tuple(samples[i])
    return samples

def as_board(splitted):
    return splitted[0] + 2 * splitted[1]


def create_table(np_func):
    orig = np.arange(81).reshape((9, 9))
    rot = np_func(orig)

    table = np.zeros(81)
    for i in range(9):
        for j in range(9):

            orig_idx = (j, i)
            orig_flat = coord_to_flat(orig_idx)

            whr = np.where(rot == orig_flat)
            to_idx = (whr[1][0], whr[0][0])
            to_flat = coord_to_flat(to_idx)

            table[orig_flat] = to_flat
    return table


def do_op(probs, table):

    out = np.zeros(82)
    for k in range(81):
        out[int(table[k])] = probs[k]

    out[81] = probs[81]

    return out


ROT_TABLE = create_table(np.rot90)
FLIP_TABLE = create_table(np.flipud)


def rot90(probs):
    global ROT_TABLE
    return do_op(probs, ROT_TABLE)


def flipud(probs):
    global FLIP_TABLE
    return do_op(probs, FLIP_TABLE)

def swap_colors(s):
    sample = deepcopy(s)
    for k in range(7):
        sample[0][2 * k], sample[0][2 * k + 1] = deepcopy(sample[0][2 * k + 1]), deepcopy(sample[0][2 * k])

    if SWAP_CURRENT:
        sample[0][14] = 1 - sample[0][14]

    if SWAP_VALUE:
        sample[2] = 1 - sample[2]
    return sample


def augmentate(samples):
    size = len(samples)
    samples = listify(samples)
    augmentated_samples = deepcopy(samples)

    for _ in range(15 if ADD_SWAP else 7):
        augmentated_samples += deepcopy(samples)

    for rot in range(1, 4):
        for idx in range(size):
            for k in range(15):
                augmentated_samples[rot * size + idx][0][k] = np.rot90(augmentated_samples[(rot - 1) * size + idx][0][k])
            augmentated_samples[rot * size + idx][1] = rot90(augmentated_samples[(rot - 1) * size + idx][1])

    for mirror in range(4, 8):
        for idx in range(size):
            for k in range(15):
                augmentated_samples[mirror * size + idx][0][k] = np.flipud(augmentated_samples[(mirror - 4) * size + idx][0][k])
            augmentated_samples[mirror * size + idx][1] = flipud(augmentated_samples[(mirror - 4) * size + idx][1])

    if ADD_SWAP:
        for swap in range(8 * size):
            augmentated_samples[swap + 8 * size] = swap_colors(augmentated_samples[swap])

    return tuplefy(augmentated_samples)

f = open('./new_samples.npy', 'rb')
samples = pickle.load(f)
f.close()

a = augmentate(samples)

f = open(f'./new_samples_aug{"" if ADD_SWAP else "_half"}.npy', 'wb')
pickle.dump(a, f)
f.close()
