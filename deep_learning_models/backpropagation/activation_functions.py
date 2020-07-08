import numpy as np


def relu(input):
    # print('relu return value: ',np.maximum(input,0))
    return np.maximum(input, 0)


def sigmoid(input):
    return 1 / (1 + np.exp(-input))


def tanh(input):
    return np.tanh(input)


def softmax_activation(input):
    num = np.exp(input)
  #  print('\n normalisaton sum: ', np.sum(num, axis=1, keepdims=True))
    softmax_output = np.exp(input) / np.sum(num, axis=1, keepdims=True)
    # print(softmax_output)
    return softmax_output

# letting the derivative be defined at 0 for safety


def derivative_relu(input):
    input[input <= 0] = 0
    input[input > 0] = 1
    # print('derivative relu: ', input)
    return input


def derivative_sigmoid(input):
    return sigmoid(input) * (1 - sigmoid(input))


def derivative_tanh(input):
    return 1 - tanh(input) * tanh(input)


def derivative(activation_function, input):
    if activation_function == 'sigmoid':
        return derivative_sigmoid(input)
    elif activation_function == 'relu':
        return derivative_relu(input)
    elif activation_function == 'tanh':
        return derivative_tanh(input)


# Subtracting the max term to insure numerical stability
# not using stable softmax at the moment
def softmax_stable(input):
    #print('\n input shape :', input.shape)

    numerator = np.zeros((input.shape[0], input.shape[1]))
    #print('shape numerator: ', numerator.shape)

    minibatch_size, _ = input.shape

    def x(row):
        new_row = np.exp(row - row.max())
        return new_row
    output = np.apply_along_axis(x, 1, input)
    row_sum = np.sum(output, axis=1).reshape((minibatch_size, 1))
    output = np.divide(output, row_sum)
  #  print('\n new softmax return: ', output)
    return output
