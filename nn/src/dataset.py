import gzip
import os
import json
import urllib.request
import numpy as np
import pickle

import torch
from torch.utils.data import TensorDataset, DataLoader

device = "cuda:0" if torch.cuda.is_available() else "cpu"


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


def name_to_coord(s):
    assert s != "PASS"
    indexLetters = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'J': 8}

    col = indexLetters[s[0]]
    lin = int(s[1:]) - 1
    return lin, col


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def augmentate_input(X, size):
    '''The X vector should be long enough for modifications to be done in place'''
    for rot in range(1, 4):
        for idx in range(size):
            X[rot*size + idx, 0] = np.rot90(X[(rot-1)*size + idx, 0])
            X[rot*size + idx, 1] = np.rot90(X[(rot-1)*size + idx, 1])
            X[rot*size + idx, 2] = np.rot90(X[(rot-1)*size + idx, 2])

    for mirror in range(4, 8):
        for idx in range(size):
            X[mirror*size + idx, 0] = np.flipud(X[(mirror-4) * size + idx, 0])
            X[mirror*size + idx, 1] = np.flipud(X[(mirror-4) * size + idx, 1])
            X[mirror*size + idx, 2] = np.flipud(X[(mirror-4) * size + idx, 2])

    return X


def get_raw_dataset(split=0.8):

    filename = 'dataset.npy'

    if os.path.isfile(filename):
        return pickle.load(open(filename, 'rb'))

    data = get_raw_data_go()
    size = len(data)

    X = np.zeros((8*size, 3, 9, 9))
    y = np.array(8*[d["black_wins"]/d["rollouts"] for d in data])

    for idx, table in enumerate(data):

        for bp in table['black_stones']:
            i, j = name_to_coord(bp)
            X[idx, 0, i, j] = 1

        for wp in table['white_stones']:
            i, j = name_to_coord(wp)
            X[idx, 1, i, j] = 1

        X[idx, 2, :, :] = len(table['list_of_moves']) % 2

    X = augmentate_input(X, size)

    X, y = unison_shuffled_copies(X, y)

    idx = int(split*size)

    X_train = X[:idx]
    y_train = y[:idx]

    X_test = X[idx:]
    y_test = y[idx:]

    pickle.dump(((X_train, y_train), (X_test, y_test)), open(filename, 'wb'))
    return ((X_train, y_train), (X_test, y_test))


def get_loader(X, y, batch_size=64):

    X = torch.Tensor(X).to(device)
    y = torch.Tensor(y).to(device)

    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size)

    return loader


def get_loaders(batch_sizes=[64, 8192]):
    ((X_train, y_train), (X_test, y_test)) = get_raw_dataset()

    train_loader = get_loader(X_train, y_train, batch_size=batch_sizes[0])
    test_loader = get_loader(X_test, y_test, batch_size=batch_sizes[1])

    return train_loader, test_loader
