import numpy as np
from com.asu.fsl.models.Data import Data
from com.asu.fsl.utilities import dataUtils
from com.asu.fsl.utilities.mathUtils import calculate_training_error
from com.asu.fsl.models.NeuralNetConfig import NeuralNetConfig

class NeuralNet(object):


    nn_config = NeuralNetConfig(4)

    def __init__(self, nn_config):
        self.nn_config = nn_config

    '''
        Organize data and setup for training the ANN
    '''
    def setup(self):
        self.setup_data()
        self.initialize_weights()
    
    def setup_data(self):
        base_path = "D:\\Projects\\Eclipse" 
        # D:\Projects\Eclipse\FSL-Phase-2\data
        #####################################################################################
        file_dict = {}
        file_dict["train_class_1"] = base_path + "\\FSL-Phase-2\\data\\Train1.txt"
        file_dict["train_class_2"] = base_path + "\\FSL-Phase-2\\data\\Train2.txt"
        file_dict["test_class_1"] = base_path + "\\FSL-Phase-2\\data\\Test1.txt"
        file_dict["test_class_2"] = base_path + "\\FSL-Phase-2\\data\\Test2.txt"
        dataUtils.load_data(file_dict)

        # Invoking singleton
        data_obj = Data()

        self.nn_config.training_data = data_obj.get_training_data()[:, :2]
        self.nn_config.training_class_data = data_obj.get_training_data()[:, 2]
        self.nn_config.testing_data = data_obj.get_testing_data()[:, :2]
        self.nn_config.testing_class_data = data_obj.get_testing_data()[:, 2]
        self.nn_config.validation_data = data_obj.get_validation_data()[:, :2]
        self.nn_config.validation_class_data = data_obj.get_validation_data()[:, 2]

    '''
        Initialize weights of the ANN as per textbook rule of thumb, 1/sqrt(no of hidden nodes)
    '''
    # def initialize_weights(self):
    #     self.nn_config.layers.clear()
    #     weights_input_hidden = np.random.uniform(-1/np.sqrt(self.nn_config.hidden_layer_size), 1/np.sqrt(self.nn_config.hidden_layer_size), size=(self.nn_config.input_layer_size+1, self.nn_config.hidden_layer_size))
    #     print("Hidden weights\n")
    #     print(weights_input_hidden)
    #     weights_hidden_output = np.random.uniform(-1/np.sqrt(self.nn_config.hidden_layer_size), 1/np.sqrt(self.nn_config.hidden_layer_size), size=(self.nn_config.hidden_layer_size+1, self.nn_config.output_layer_size))
    #     print("Output weights\n")
    #     print(weights_hidden_output)
    #     self.nn_config.layers.append({"bias": weights_input_hidden})   
    #     self.nn_config.layers.append({"bias": weights_hidden_output})

    def initialize_weights(self):
        self.nn_config.layers.clear()
        weights_input_hidden = np.random.uniform(-1/np.sqrt(self.nn_config.hidden_layer_size), 1/np.sqrt(self.nn_config.hidden_layer_size), size=(self.nn_config.input_layer_size+1, self.nn_config.hidden_layer_size))
        print("Hidden weights\n")
        print(weights_input_hidden)
        weights_hidden_output = np.random.uniform(-1/np.sqrt(self.nn_config.hidden_layer_size), 1/np.sqrt(self.nn_config.hidden_layer_size), size=(self.nn_config.hidden_layer_size+1, self.nn_config.output_layer_size))
        print("Output weights\n")
        print(weights_hidden_output)
        self.nn_config.layers.append({"bias": weights_input_hidden})
        self.nn_config.layers.append({"bias": weights_hidden_output})



    def get_activation_fn_op(self, val):
        return val if val > 0  else 0

    # def fwd_prop(self, sample):
    #     sample = np.hstack((1, sample[:]))
    #     print("ANN sample:", sample.transpose().shape)
    #     for layer_index in range(len(self.nn_config.layers)):
    #         outputs = []
    #         self.nn_config.layers[layer_index]["outputs"] = []
    #         for node_index in range(len(self.nn_config.layers[layer_index])):
    #             print("Bias shape:", self.nn_config.layers[layer_index]["bias"].shape)
    #             self.nn_config.layers[layer_index]["output"] = self.get_activation_fn_op(np.dot(sample.transpose(),self.nn_config.layers[layer_index]["bias"]))
    #             outputs.append(self.nn_config.layers[layer_index]["output"])
    #         inputs = np.hstack(([1], outputs[:]))
    #     print("Inputs: ", inputs[1:])
    #     return inputs[1:]

    def fwd_prop(self, sample):
        sample = np.hstack((1, sample[:]))
        print("ANN sample:", sample.transpose().shape)
    
        for layer_index in range(len(self.nn_config.layers)):
            outputs = []
            self.nn_config.layers[layer_index]["outputs"] = []
    
            for node_index in range(self.nn_config.layers[layer_index]["bias"].shape[1]):
                # Use the dot product to calculate the weighted sum of inputs
                weighted_sum = np.dot(sample, self.nn_config.layers[layer_index]["bias"][:, node_index])
    
                # Apply the activation function to the weighted sum
                output = self.get_activation_fn_op(weighted_sum)
    
                # Store the output for further use
                self.nn_config.layers[layer_index]["outputs"].append(output)
    
            # Update the sample for the next layer
            sample = np.hstack(([1], self.nn_config.layers[layer_index]["outputs"][:]))
    
        print("Final Outputs: ", sample[1:])
        return sample[1:]


    def train(self):

        epoch = 0
        learning_rate = 10e-3


        training_error_history = []
        validation_error = float('inf')
        validation_error_history = []
        testing_error_history = []

        stopping_condition_met = False
        while not stopping_condition_met:
            error = 0.0
            for index, sample in enumerate(self.nn_config.training_data):
                output = self.fwd_prop(sample)
                error += calculate_training_error(output, self.nn_config.training_class_data[index])
                self.back_prop(self.nn_config.training_class_data[index])
                self.__update_weights(sample, learning_rate)

            training_error = training_error/self.training_data.shape[0]
            updated_val_error = self.__get_validation_error()
            testing_error = self.__get_testing_error()

            stopping_condition_met = np.isclose(validation_error, updated_val_error) or epoch >= n_epochs
            validation_error = updated_val_error
            print("> epoch: {}; learning_rate: {}; training_error: {}; validation_error: {}"
                  .format(epoch, learning_rate, training_error, validation_error))
            training_error_history.append(training_error)
            validation_error_history.append(validation_error)
            testing_error_history.append(testing_error)
            
            epoch += 1
            
        print("Finished training")
        print()


    
def train_network(ann):
    # first setup all of the data
    ann.setup()
    ann.train()

if __name__ == '__main__':
    avg_scores = []
    nn_config = NeuralNetConfig(4)
    for no_hidden in range(4, 15, 2):
        nn_config.hidden_layer_size = no_hidden
        scores = []
        for _ in range(5):
            # ann -Artificial Neural Network
            ann = NeuralNet(nn_config)
            score = train_network(ann)
            scores.append(score)
        avg_scores.append(np.mean(scores))

    with open('scores.txt', 'w') as f:
        f.write(str(avg_scores))