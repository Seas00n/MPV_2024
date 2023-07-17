import numpy as np


def fifo_data_vec(data_mat, data_vec):
    data_mat[:-1] = data_mat[1:]
    data_mat[-1] = data_vec
    return data_mat
