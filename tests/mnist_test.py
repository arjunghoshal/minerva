import os
from mnist_loader import load_data_wrapper
from minerva.neural_network import NeuralNetwork


def test_mnist():
    print(os.getcwd())
    training_data, validation_data, test_data = load_data_wrapper("tests/mnist.pkl.gz")
    net = NeuralNetwork([784, 30, 10])
    net.train(training_data, 30, 10, 3.0, test_data)
