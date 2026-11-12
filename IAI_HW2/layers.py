import abc
from typing import List

import numpy as np


class Layer(abc.ABC):
    @abc.abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propagate through the layer.

        Parameters:
            x (`np.ndarray`):
                input tensor with shape `(batch_size, input_dim)`.

        Returns:
            out (`np.ndarray`):
                output tensor of the layer with shape
                `(batch_size, output_dim)`.
        """
        return NotImplemented

    @abc.abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward propagate through the layer.

        Calculate the loss gradient with respect to the parameters of the
        layer and the gradient with respect to the input of the layer. The
        gradients with respect to the parameters are summed over the batch.

        Parameters:
            grad (`np.ndarray`):
                gradient with respect to the output of the layer.

        Returns:
            grad (`np.ndarray`):
                gradient with respect to the input of the layer.
        """
        return NotImplemented

    def params(self) -> List[np.ndarray]:
        """Return the parameters of the layer.

        The order of parameters should be the same as the order of gradients
        returned by `self.grads()`.
        """
        return []

    def grads(self) -> List[np.ndarray]:
        """Return the gradients of the layer.

        The order of gradients should be the same as the order of parameters
        returned by `self.params()`.
        """
        return []


class Sequential(Layer):
    """A sequential model that stacks layers in a sequential manner.

    Parameters:
        layers (`List[Layer]`):
            list of layers to be added to the model.
    """
    def __init__(self, *layers: List[Layer]):
        self.layers = []
        self.append(*layers)

    def append(self, *layers: List[Layer]):
        """Add layers to the model"""
        self.layers.extend(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Sequentially forward propagation through layers.

        Parameters:
            x (`np.ndarray`):
                input tensor.

        Returns:
            out (`np.ndarray`):
                output tensor of the last layer.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Sequentially backward propagation through layers.

        Parameters:
            grad (`np.ndarray`):
                gradient of loss function with respect to the output of forward
                propagation.

        Returns:
            grad (`np.ndarray`):
                gradient with respect to the input of forward propagation.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self) -> List[np.ndarray]:
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.params())
        return parameters

    def grads(self) -> List[np.ndarray]:
        gradients = []
        for layer in self.layers:
            gradients.extend(layer.grads())
        return gradients


class MatMul(Layer):
    """Matrix multiplication layer.

    Parameters:
        W (`np.ndarray`):
            weight matrix with shape `(input_dim, output_dim)`.
    """
    def __init__(self, W: np.ndarray):
        self.W = W

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        return x @ self.W

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.W_grad = self.cache.T @ grad
        return grad @ self.W.T

    def params(self) -> List[np.ndarray]:
        return [self.W]

    def grads(self) -> List[np.ndarray]:
        return [self.W_grad]


class Bias(Layer):
    """Bias layer.

    Parameters:
        b (`np.ndarray`):
            bias vector with shape `(output_dim,)`.
    """
    def __init__(self, b: np.ndarray):
        self.b = b

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.b_grad = np.sum(grad, axis=0)
        return grad

    def params(self) -> List[np.ndarray]:
        return [self.b]

    def grads(self) -> List[np.ndarray]:
        return [self.b_grad]


class ReLU(Layer):
    """Rectified Linear Unit."""

    def forward(self, x):
        self.cache = x
        return np.clip(x, 0, None)

    def backward(self, grad):
        x = self.cache
        return np.where(x > 0, grad, 0)


class Sigmoid(Layer):
    """Sigmoid function."""

    def forward(self, x):
        z = np.exp(-np.abs(x))
        out = np.where(x >= 0, 1 / (1 + z), z / (1 + z))
        self.cache = out
        return out

    def backward(self, grad):
        out = self.cache
        return grad * out * (1 - out)


class Softmax(Layer):
    """Softmax function over the last axis."""

    def forward(self, x):
        z = x - np.max(x, axis=-1, keepdims=True)
        exp_z = np.exp(z)
        out = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
        self.cache = out
        return out

    def backward(self, grad):
        out = self.cache
        return out * (grad - np.sum(grad * out, axis=-1, keepdims=True))
