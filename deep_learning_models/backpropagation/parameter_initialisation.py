import numpy as np
np.random.seed(42)

def bias_initialisation(n_hidden, dims):
    bias = []
    for i in range(n_hidden + 1):
        bias.append(np.zeros((1, dims[i + 1])))

    return bias


def weight_initial_zeros(n_hidden, dims):
    weights = []
    for i in range(n_hidden + 1):
        w_temp = np.zeros((dims[i], dims[i + 1]))
        weights.append(w_temp)
    return weights

# check weight dimensions
def weight_initial_normal(n_hidden, dims):
    weights = []
    for i in range(n_hidden + 1):
        w_temp = np.random.normal(0, 1, (dims[i], dims[i + 1]))
        weights.append(w_temp)
    return weights


# dims=[784, ,, 10]

# hL-1 number of neurons feeding in and hL , number of neurons in this particular layer
def weight_initial_glorot(n_hidden, dims):
    weights = []
    for i in range(n_hidden + 1):
        t_temp = 6 / (dims[i] + dims[i + 1])
        d_temp = np.sqrt(t_temp)
        w_temp = np.random.uniform(-d_temp, d_temp, (dims[i], dims[i + 1]))
        weights.append(w_temp)

    return weights

