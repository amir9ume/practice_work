import os

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset

from problem1.activation_functions import *
from problem1.data_load import mnist_testset, mnist_trainset, one_hot_encoding
from problem1.parameter_initialisation import *
import pickle
from sklearn.metrics import accuracy_score

# for range of parameters in .5M to 1M, upper limit (1000,300). lower limit (500,200)
# (800,300) safe size


train_loss, validation_loss, test_loss = [], [], []
train_accuracy, validation_accuracy, test_accuracy = [], [], []


class NN(object):

    # datapath.
    # Why mode argument here??
    def __init__(self, dimensions, learning_rate, epochs,
                 n_hidden=2, mode='train',
                 datapath=None, model_path=None):

        self.n_hidden = n_hidden
        self.dimensions = dimensions
        self.learning_rate = learning_rate
        self.epochs = epochs

    # zero intialisation, gaussian or glorot
    def initialize_weights(self, initialisation_type):
        assert initialisation_type == 'zeros' or initialisation_type == 'normal' \
               or initialisation_type == 'glorot'

        bias = bias_initialisation(self.n_hidden, self.dimensions)

        # they have to be unequal sized
        if (initialisation_type == 'zeros'):
            weights_hidden1, weights_hidden2, weights_hidden3 = weight_initial_zeros(
                self.n_hidden, self.dimensions)

        if (initialisation_type == 'normal'):
            weights_hidden1, weights_hidden2, weights_hidden3 = weight_initial_normal(
                self.n_hidden, self.dimensions)

        # cross check definition of glorot
        if (initialisation_type == 'glorot'):
            weights_hidden1, weights_hidden2, weights_hidden3 = weight_initial_glorot(
                self.n_hidden, self.dimensions)

        weights = [weights_hidden1, weights_hidden2, weights_hidden3]

        return weights, bias

    def forward(self, input, weights, bias, activation_functions):

        preactivation_layer1 = np.matmul(input, weights[0]) + bias[0]
        output_layer_1 = self.activation(
            preactivation_layer1, activation_functions['layer1'])

        preactivation_layer2 = np.matmul(output_layer_1, weights[1]) + bias[1]
        output_layer_2 = self.activation(
            preactivation_layer2, activation_functions['layer2'])

        preactivation_softmax_layer = np.matmul(
            output_layer_2, weights[2]) + bias[2]

        #      print('preactivation layer3: ', preactivation_softmax_layer)
        output_layer_3 = self.softmax(preactivation_softmax_layer)

        #        print('shape output layer3 : ', output_layer_3.shape)
        #       print('output layer3 classes: ', output_layer_3[0])
        #      print('output layer3 data type: ', output_layer_3.dtype)
        prediction_classes = np.argmax(output_layer_3, axis=1)

        prediction_classes = prediction_classes.reshape(-1, 1)

        layer_wise_output = {'layer1': output_layer_1, 'layer2': output_layer_2, 'layer3': output_layer_3,
                             'prediction': prediction_classes, 'affine1': preactivation_layer1,
                             'affine2': preactivation_layer2}

        return layer_wise_output

    # choose between relu, sigmoid, tanh
    def activation(self, input, activation_type):
        assert activation_type == 'relu' \
               or activation_type == 'sigmoid' or 'tanh'

        if (activation_type == 'relu'):
            a = relu(input)

        elif (activation_type == 'sigmoid'):
            a = sigmoid(input)

        elif (activation_type == 'tanh'):
            a = tanh(input)

        return a

    def cross_entropy(self, target_one_hot_encoded, output_softmax):
        return -np.log(output_softmax[np.where(target_one_hot_encoded)])

    # def loss(self, prediction, target, mini_batch_size):
    def loss(self, layer_wise_output, target, mini_batch_size):

        target = one_hot_encoding(target)

        average_loss = np.mean(
            [self.cross_entropy(target, layer_wise_output['layer3'])])

        # print('\n prediction: ', np.argmax(layer_wise_output['prediction'], axis=1), ": vs actual: ", np.argmax(target, axis=1))

        return average_loss

    # Using stable softmax
    def softmax(self, input):
        return softmax_stable(input)

    # pass all useful previous values through cache

    def backward(self, cache, labels, mini_batch_size, activation_functions):
        # derivative is --> dLoss/dLayer3_i = P_i - C_i. c is one hot encoded, p is probabilities
        weights, bias, layer_wise_outputs, image_data = cache

        # trying one hot encoding
        labels = one_hot_encoding(labels)

        derivative_wrt_preactivation_layer3 = (
                                                      layer_wise_outputs['layer3'] - labels) / labels.shape[0]
        derivative_wrt_weights_hidden3 = np.matmul(
            layer_wise_outputs['layer2'].T, derivative_wrt_preactivation_layer3)

        derivative_wrt_bias_hidden3 = np.sum(
            derivative_wrt_preactivation_layer3, axis=0, keepdims=True)

        dg = derivative_wrt_preactivation_layer3

        df = np.matmul(weights[2], derivative_wrt_preactivation_layer3.T)

        #   derivative_activation_function_layer2 = derivative(activation_functions['layer2'], layer_wise_outputs['layer2'])
        derivative_activation_function_layer2 = derivative(activation_functions['layer2'],
                                                           layer_wise_outputs['affine2'])

        de = np.multiply(df, derivative_activation_function_layer2.T)

        dd = de
        dc = np.matmul(weights[1], dd)
        derivative_wrt_bias_hidden2 = np.sum(de.T, axis=0, keepdims=True)
        derivative_wrt_weights_hidden2 = (
            np.matmul(dd, layer_wise_outputs['layer1']))

        derivative_activation_function_layer1 = derivative(activation_functions['layer1'],
                                                           layer_wise_outputs['affine1'])

        db = np.multiply(derivative_activation_function_layer1, dc.T)
        da = db

        derivative_wrt_weights_hidden1 = np.matmul(image_data.T, da)
        derivative_wrt_bias_hidden1 = np.sum(db, axis=0, keepdims=True)

        gradients_weights = [derivative_wrt_weights_hidden1, derivative_wrt_weights_hidden2.T,
                             derivative_wrt_weights_hidden3]
        gradients_bias = [derivative_wrt_bias_hidden1,
                          derivative_wrt_bias_hidden2, derivative_wrt_bias_hidden3]

        return gradients_weights, gradients_bias

    # update rule
    def update(self, grads_weights, grads_bias, weights, bias):
        for i in range(len(weights)):
            weights[i] -= np.multiply(self.learning_rate, grads_weights[i])
            bias[i] -= np.multiply(self.learning_rate, grads_bias[i])
        return weights, bias

    def train(self, mnist_trainset, mini_batch_size, activation_functions, initialisation_type):

        data_loader_trainset = DataLoader(
            mnist_trainset, batch_size=mini_batch_size, shuffle=True)

        weights, bias = self.initialize_weights(initialisation_type)

        predictions_train_list=[]
        targets_train_list=[]

        iterations = 0
        for epoch in range(self.epochs):

            for _, (inputs, targets) in enumerate(data_loader_trainset):
                inputs = inputs.numpy()
                targets = targets.numpy()
                inputs = inputs.reshape(inputs.shape[0], 784).astype('float32')



                # forward propagation
                layer_wise_outputs = self.forward(
                    inputs, weights, bias, activation_functions)

                average_loss = self.loss(
                    layer_wise_outputs, targets, mini_batch_size)
                train_loss.append(average_loss)

                cache = weights, bias, layer_wise_outputs, inputs

                # pass the actual arguments
                gradients_weights, gradients_bias = self.backward(
                    cache, targets, mini_batch_size, activation_functions)

                weights, bias = self.update(
                    gradients_weights, gradients_bias, weights, bias)

                pred = layer_wise_outputs['prediction']
                predictions_train_list.extend(pred.reshape(-1))
                targets_train_list.extend(targets)

                #iterations += 1
                #if iterations % 100 == 0:
                #    print('average loss training at iteration: ',
                #          iterations, '::', average_loss)

            print('------------------------------------')
            print('\n Epoch: ', epoch, )
            print('Average training loss at end of epoch: ',
                  epoch , '::', average_loss)

            print(accuracy_score(targets_train_list, predictions_train_list, normalize=True))


            self.test(weights,bias,activation_functions,mini_batch_size)


            weights_copy= np.copy(weights)

#            N= self.N_calculate(3,3)
            #epsilon= 1/10
            #print('epsilon', epsilon)
            #print('gradient numerator is',self.loss_function_plus_minus(layer_wise_outputs,weights_copy, epsilon, activation_functions))



        trained_parameters = {'weights': weights, 'bias': bias}
        pickle.dump(trained_parameters, open("trained.p", "wb"))


        return weights, bias, activation_functions

    def test(self, trained_weights, trained_bias, activaton_functions, mini_batch_size):
        #mini_batch = mini_batch_size
        mini_batch=1
        data_loader_testset = DataLoader(
            mnist_testset, batch_size=mini_batch, shuffle=False)
        # for validation set
#        print('size validation: ', len(data_loader_testset))

        predictions_list = []
        validation_targets_list = []

        counter = 0
        for _, (validation_data, validation_targets) in enumerate(data_loader_testset):
            validation_data = validation_data.numpy()
            validation_targets = validation_targets.numpy()

            validation_data = validation_data.reshape(
                validation_data.shape[0], 784).astype('float32')

            layer_wise_outputs = self.forward(
                validation_data, trained_weights, trained_bias, activaton_functions)

            # print('shape prediction: ', layer_wise_outputs['prediction'].shape)
            # print('shape validation_targets: ', validation_targets.shape)

            average_loss_validation = self.loss(
                layer_wise_outputs, validation_targets, mini_batch)
            #            z = np.sum(layer_wise_outputs['prediction'] == validation_targets)

            #            validation_accuracy = z / len(validation_targets)

            # print("validation targets: ", validation_targets)
            pred = layer_wise_outputs['prediction']
            predictions_list.extend(pred[0].reshape(-1))
            validation_targets_list.extend(validation_targets)
            # print("predictions : ",pred[0])

            # print('accuracy: ', np.mean(layer_wise_outputs['prediction']==validation_targets))
            validation_loss.append(average_loss_validation)
            counter += 1

        # print('predictions: ', predictions_list)
        # print('validation list: ', validation_targets_list)
        validation_accuracy = accuracy_score(validation_targets_list, predictions_list)
        print("validation accuracy: ", validation_accuracy)


    def loss_function_plus_minus(y_target, layer_wise_outputs, weights, epsilon, activation_functions):
        # check this formula correctness
        weights_layer_2 = weights[1]
        weights_layer_3 = weights[2]

        plus_fx_loss=[]
        minus_fx_loss=[]

        for i in range(10):
            temp= weights_layer_2[i]
            weights_layer_2[i] += epsilon
            plus_fx_affine_layer2= np.matmul(weights_layer_2, layer_wise_outputs['layer1'])
            plus_fx_layer2 = self.activation(
            plus_fx_affine_layer2, activation_functions['layer2'])

            plus_fx_Output = self.softmax(plus_fx_layer2)

            plus_fx_loss.extend(self.loss_average_output_layer (plus_fx_Output, y_target, 1))

            weights_layer_2[i]= temp
            weights_layer_2[i] -= epsilon
            minus_fx_affine_layer2= np.matmul(weights_layer_2, layer_wise_outputs['layer1'])
            minus_fx_layer2 = self.activation(
            layer_wise_outputs['affine2'], activation_functions['layer2'])

            minus_fx_Output = self.softmax(minus_fx_layer2)

            minus_fx_loss.extend(self.loss_average_output_layer(minus_fx_Output, y_target,1))

        print('fx plus loss: ', plus_fx_loss)
        print('fx minus loss: ', minus_fx_loss)
        return plus_fx_loss, minus_fx_loss, epsilon

    def loss_average_output_layer(self, output_layer_3, target):

        target = one_hot_encoding(target)

        average_loss = np.mean(
            [self.cross_entropy(target, output_layer_3)])

        # print('\n prediction: ', np.argmax(layer_wise_output['prediction'], axis=1), ": vs actual: ", np.argmax(target, axis=1))

        return average_loss

    def grad_calculate(loss_epsilon_plus, loss_epsilon_minus, epsilon):
        grad = (loss_epsilon_plus - loss_epsilon_minus) / 2(epsilon)
        return grad

    def N_calculate(k, i):
        return k * (10 ** i)
