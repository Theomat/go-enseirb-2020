import json
import numpy as np
from GnuGo import *
from tqdm import tqdm
import pickle
import urllib.request
import os
import gzip


def get_raw_data_go():
    ''' Returns the set of samples from the local file or download it if it does not exists'''

    raw_samples_file = "samples-9x9.json.gz"

    if not os.path.isfile(raw_samples_file):
        print("File", raw_samples_file, "not found, I am downloading it...", end="")
        urllib.request.urlretrieve("https://www.labri.fr/perso/lsimon/ia-inge2/samples-9x9.json.gz", "samples-9x9.json.gz")
        print(" Done")

    with gzip.open("samples-9x9.json.gz") as fz:
        data = json.loads(fz.read().decode("utf-8"))
    return data


indexLetters = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'J': 8}


def name_to_coord(name):
    if name == 'PASS':
        return (-1, -1)

    col = indexLetters[name[0]]
    lin = int(name[1:]) - 1

    return (col, lin)


def coord_to_flat(coord):
    if coord == (-1, -1):
        return 81
    return 9 * coord[1] + coord[0]


def name_to_flat(name):
    return coord_to_flat(name_to_coord(name))


def get_prob_reward(table, gnugo):

    moves = gnugo.Moves(gnugo)

    for move in table['list_of_moves']:
        moves.playthis(move)

    status, _ = moves._gnugo.query("experimental_score " + moves._nextplayer)

    if status != "OK":
        return None, None

    status, possible_moves = moves._gnugo.query("top_moves " + moves._nextplayer)

    possible_moves = possible_moves.strip().split()

    if len(possible_moves) == 0:
        return None, None

    best_moves = [m for idx, m in enumerate(possible_moves) if idx % 2 == 0]
    scores = np.array([float(s) for idx, s in enumerate(possible_moves) if idx % 2 == 1])

    assert len(best_moves) == len(scores)

    prob_distr = scores / scores.sum()
    probs = np.zeros(82)

    for idx, m in enumerate(best_moves):
        flat_move = name_to_flat(m)
        probs[flat_move] = prob_distr[idx]

    if table['depth'] % 2 == 0:                         # black plays next
        reward = table['black_wins'] / table['rollouts']
    else:                                               # white plays next
        reward = table['white_wins'] / table['rollouts']

    gnugo.query("clear_board")

    return probs, reward


history_size = 7
samples = []

gnugo = GnuGo(9)
tables = get_raw_data_go()

for idx, table in enumerate(tqdm(tables)):

    assert table['depth'] == len(table['list_of_moves'])

    vector = np.zeros((2 * history_size + 1, 9, 9), dtype=np.float64)
    base = np.zeros((2, 9, 9), dtype=np.float64)
    next_to_play = 0
    skip = False

    for move in table['list_of_moves'][:-history_size]:

        (col, lin) = name_to_coord(move)

        if base[next_to_play, lin, col] != 0:  # We wont analyse games with captures for now
            skip = True

        base[next_to_play, lin, col] = 1

        next_to_play = (next_to_play + 1) % 2

    for idx, move in enumerate(table['list_of_moves'][-history_size:]):

        (col, lin) = name_to_coord(move)

        if base[next_to_play, lin, col] != 0:
            skip = True

        base[next_to_play, lin, col] = 1

        real_size = len(table['list_of_moves'][-history_size:])
        final_idx = 2 * (real_size - 1 - idx)

        vector[final_idx + 1] = base[1]
        vector[final_idx] = base[0]

        next_to_play = (next_to_play + 1) % 2

    vector[2 * history_size] = table['depth'] % 2

    if not skip:
        p, r = get_prob_reward(table, gnugo)

        if p is None and r is None:
            continue

        samples.append((vector, p, r))

f = open('./samples.npy', 'wb')
pickle.dump(samples, f)
f.close()
