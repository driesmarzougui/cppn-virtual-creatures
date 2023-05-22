import math
import numba as nb
import numpy as np


def get_activation_function(name: str):
    if name == "bss":
        return bss
    elif name == "sigmoid":
        return sigmoid
    elif name == "abs":
        return abs_f
    elif name == "nabs":
        return neg_abs_f
    elif name == "gauss":
        return gauss
    elif name == "sin":
        return sin
    elif name == "cos":
        return cos
    elif name == "identity":
        return identity
    elif name == "tanh":
        return tanh
    elif name == "square":
        return square
    elif name == "nsquare":
        return neg_square
    elif name == "relu":
        return relu


@nb.njit()
def bss(x):
    # Bipolar Steepened Sigmoid
    return 2.0 / (1 + math.exp(-x)) - 1.0


@nb.njit()
def gauss(z):
    z = max(-3.4, min(3.4, z))
    return math.exp(-5.0 * z ** 2)


@nb.njit()
def tanh(z):
    z = max(-60.0, min(60.0, 2.5 * z))
    return math.tanh(z)


@nb.njit()
def abs_f(z):
    return abs(z)


@nb.njit()
def neg_abs_f(z):
    return -abs(z)


@nb.njit()
def identity(x):
    return x


@nb.njit()
def sin(x):
    x = max(-60.0, min(60.0, 5.0 * x))
    return math.sin(x)


@nb.njit()
def cos(x):
    return np.cos(x)


@nb.njit()
def sigmoid(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return 1.0 / (1.0 + math.exp(-z))


@nb.njit()
def square(z):
    return z ** 2


@nb.njit()
def neg_square(z):
    return -(z ** 2)


@nb.njit()
def relu(z):
    return max(z, 0)
