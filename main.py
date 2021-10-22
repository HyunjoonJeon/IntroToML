import numpy as np


def load_dataset(filepath):
    return np.loadtxt(filepath)


arr = load_dataset("wifi_db/clean_dataset.txt")
print(arr)
