from problem1.data_load import mnist_trainset
from problem1.network_model import NN
import pickle

initialisations = ['zeros', 'normal', 'glorot']
activation_functions = {'layer1': 'relu', 'layer2': 'sigmoid'}


def main():
    learning_rate = 0.01
    dims = [784, 800, 300, 10]
    epochs= 10
    nn = NN(dims, learning_rate, epochs )
    mini_batch_size = 100

    weights, bias, activation_functions_trained = nn.train(
     mnist_trainset, mini_batch_size, activation_functions, initialisations[2])

    train_param = pickle.load(open("trained.p", "rb"))
    weights1= train_param['weights']
    bias1= train_param['bias']


    nn.test(weights1, bias1, activation_functions, mini_batch_size)
    #nn.plot_results()



if __name__ == '__main__':
    main()
