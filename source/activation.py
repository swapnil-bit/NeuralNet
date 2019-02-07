import numpy as np
from abc import abstractmethod


class Activation:

    @abstractmethod
    def function(self, z: np.array) -> np.array:
        pass

    @abstractmethod
    def derivative(self, z: np.array) -> np.array:
        pass


class Sigmoid(Activation):
    def function(self, z: np.array) -> np.array:
        return 1.0 / (1.0 + np.exp(-z))

    def derivative(self, z: np.array) -> np.array:
        return self.function(z) * (1 - self.function(z))


class LinearDouble(Activation):
    def function(self, z: np.array) -> np.array:
        return 2 * z

    def derivative(self, z: np.array) -> np.array:
        return np.full(z.shape, 2)


