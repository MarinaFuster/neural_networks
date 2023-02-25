import json
import utilities.utils as u
import utilities.constants as c
import utilities.activation as a

class Perceptron:
    def __init__(self, json_configuration):
        configuration = json.load(json_configuration)

        if json_configuration:
            self.configuration = configuration
            self.weights = u.initialize_weights(configuration[c.WEIGHTS_INITIALIZATION])
            self.activation_function = a.activation_functions[configuration[c.ACTIVATION_FUNCTION]]
        else:
            raise FileNotFoundError(c.INITIAL_CONFIG_FAILURE)

    def save_weights(self, filename):
        u.save_matrix(self.weights, filename)

    def compute(self, stimuli):
        # weights.T it's 1 x (features + bias) and stimuli is (features + bias) x quantity of stimuli
        return self.activation_function(self.weights.T.matmul(stimuli))

    def __str__(self):
        description = f"Weights: {self.weights.shape}\n"
        description += f"Activation function: {self.configuration[c.ACTIVATION_FUNCTION]}"