import numpy as np

#where to put a shuffle on data here
def sgd(loss, data, weights):
    #training batch will sample over our training data
    mini_batch= training_batch(data, mini_batch_size)
    #calc_gradient

    w=w-alpha*calc_gradient
