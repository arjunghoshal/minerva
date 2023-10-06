import numpy as np
from numpy import ndarray
import random
from .activation_function import ActivationFunction, SigmoidFunction


class NeuralNetwork:
    def __init__(
        self,
        layers: list[int],
        activation_function: type[ActivationFunction] = SigmoidFunction,
    ) -> None:
        self.layers: list[int] = layers
        self.biases: list[ndarray] = []
        self.weights: list[ndarray] = []
        self.activation_function: type[ActivationFunction] = activation_function
        self.init_network()

    def init_network(self):
        self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]
        self.weights = [
            np.random.randn(y, x) for x, y in zip(self.layers[:-1], self.layers[1:])
        ]

    def feed_forward(self, input: ndarray) -> tuple[list[ndarray], list[ndarray]]:
        activations: list[ndarray] = [input]
        zs: list[ndarray] = []
        for w, b in zip(self.weights, self.biases):
            zs.append(w.dot(activations[-1]) + b)
            activations.append(self.activation_function.activate(zs[-1]))
        return activations, zs

    def back_propagation(self, input: ndarray, desired: ndarray):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activations, zs = self.feed_forward(input)
        delta = (activations[-1] - desired) * self.activation_function.prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for m in range(2, len(self.layers)):
            z = zs[-m]
            prime = self.activation_function.prime(z)
            delta = np.dot(self.weights[-m + 1].transpose(), delta) * prime
            nabla_b[-m] = delta
            nabla_w[-m] = np.dot(delta, activations[-m - 1].transpose())
        return (nabla_b, nabla_w)

    def run_batch(self, batch: list[tuple[ndarray, ndarray]], alpha: float):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.back_propagation(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [
            w - (alpha / len(batch)) * nw for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (alpha / len(batch)) * nb for b, nb in zip(self.biases, nabla_b)
        ]

    def evaluate_accuracy(self, test_data: list[ndarray]):
        test_results = [
            (np.argmax(self.feed_forward(x)[0][-1]), y) for (x, y) in test_data
        ]
        return sum(int(x == y) for (x, y) in test_results)

    def train(
        self,
        data: list[tuple],
        epochs: int,
        batch_size: int,
        alpha: float,
        test_data: list[tuple] = None,
    ):
        n = len(data)
        for j in range(epochs):
            random.shuffle(data)
            batches = [data[k : k + batch_size] for k in range(0, n, batch_size)]
            for batch in batches:
                self.run_batch(batch, alpha)
            if test_data:
                n_test = len(test_data)
                print(
                    "Epoch {0}: {1} / {2}".format(
                        j, self.evaluate_accuracy(test_data), n_test
                    )
                )
                continue
            print("Epoch {0} complete".format(j))
