
class NeuralNetConfig(object):
    
    # ANN structure
    input_layer_size = 2
    hidden_layer_size = 4
    output_layer_size = 1

    layers = []

    # Data variables
    training_data = None
    training_class_data = None
    validation_data = None
    validation_class_data = None
    testing_data = None
    testing_class_data = None

    def __init__(self, hidden_layer_size):
        self.hidden_layer_size = hidden_layer_size