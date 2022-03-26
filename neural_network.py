import numpy as np
import tensorflow as tf
import matplotlib.pylot as plt

from util import get_normalized_data, y2indicator

def error_rate(p, t):
    return np.mean(p != t)


def main():
    X, Y = get_normalized_data()


if __name__ == '__main__':
    main()