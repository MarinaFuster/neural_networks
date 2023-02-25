import constants as c
import numpy as np

def relu(x: np.array):
    return np.maximum(x, 0)

activation_functions = {
    c.STEP : np.heaviside,
    c.LINEAR : np.linear,
    c.SIGMOID : np.log,
    c.HYPERBOLIC_TANGENT : np.tanh,
    c.RELU : relu
}