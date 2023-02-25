import json
import numpy as np
import constants as c

def load_matrix(matrix_file):
    return json.load(matrix_file)

def save_matrix(matrix: np.array, filename):
    json.dump(matrix, filename)

# initializes with random float numbers between 0 and 1
def initialize_random_weights(dimensions):
    return np.random.rand(dimensions)

weights_initialization_methods = {
    c.RANDOM_METHOD: initialize_random_weights
}

# process of loading existing matrix from file or initializing according to method defined in configuration
# TODO: not ready for more layers
def initialize_weights(w_configuration):
    if (w_configuration[c.WEIGHTS_FILE]) is not None:
        return load_matrix(w_configuration[c.WEIGHTS_FILE])

    w_dimensions = w_configuration[c.FEATURES] + 1 if w_configuration[c.BIAS] else w_configuration[c.FEATURES]

    return weights_initialization_methods[w_configuration[c.METHOD]](w_dimensions)
