from numpy import ndarray, exp
from abc import ABC, abstractmethod


class ActivationFunction(ABC):
    @staticmethod
    @abstractmethod
    def activate(input: ndarray):
        pass

    @staticmethod
    @abstractmethod
    def prime(input: ndarray):
        pass


class SigmoidFunction(ActivationFunction):
    @staticmethod
    def activate(input: ndarray):
        return 1 / (1 + exp(-input))

    @staticmethod
    def prime(input: ndarray):
        return SigmoidFunction.activate(input) * (1 - SigmoidFunction.activate(input))
