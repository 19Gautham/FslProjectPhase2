# -*- coding: utf-8 -*-

from com.asu.fsl.models.Data import Data
import numpy as np
from com.asu.fsl.utilities.mathUtils import normalize_data

def load_data(file_dict):
    data_dict = {}
    data_obj = Data()
    for key in file_dict:
        print("\n Key: ", key)
        class_col = None
        data = load_file_as_array(file_dict['train_class_1'])
        data = normalize_data(data)
        if key.endswith("1"):
            class_category = 0
        else:
            class_category = 1
        if class_category == 0:
            class_col = np.zeros((data.shape[0], 1), dtype='int')
        else:
            class_col = np.ones((data.shape[0], 1), dtype='int')
        # print("H-stack: \n", np.hstack((data, class_col)))
        data_dict[key] = np.hstack((data, class_col))
        print("Data:\n", data_dict[key])

    data_obj.set_training_data(np.vstack((data_dict["train_class_1"][:1500, :], data_dict["train_class_2"][:1500, :])))
    data_obj.set_validation_data(np.vstack((data_dict["train_class_1"][1500:, :], data_dict["train_class_2"][1500:, :])))
    data_obj.set_testing_data(np.vstack((data_dict["test_class_1"], data_dict["test_class_2"])))

def load_file_as_array(file_path):

    elements = []
    with open(file_path) as f:
        for line in f.readlines():
            # converting type of data U9 to float
            x1, x2 = map(float, line.strip('\n').split("  "))
            elements.append((x1, x2))
    # a numpy array of tuples to ease data manipulation
    print(np.array(elements))
    return np.array(elements)

if __name__ == "__main__":

    # Dictionary to store data files
    # Please replace the file path "D:\\Projects\\Eclipse" here with the appropriate path
    base_path = "D:\\Projects\\Eclipse" 
    # D:\Projects\Eclipse\FSL-Phase-2\data
    #####################################################################################
    file_dict = {}
    file_dict["train_class_1"] = base_path + "\\FSL-Phase-2\\data\\Train1.txt"
    file_dict["train_class_2"] = base_path + "\\FSL-Phase-2\\data\\Train2.txt"
    file_dict["test_class_1"] = base_path + "\\FSL-Phase-2\\data\\Test1.txt"
    file_dict["test_class_2"] = base_path + "\\FSL-Phase-2\\data\\Test2.txt"
    
    load_data(file_dict)
    
    # data = load_file_as_array(file_dict['train_class_1'])
    # print(data)
    # data = normalize_data(data)
    # print(data)
    #
    # print("Y:\n",y)
    # print("Hstack:\n", numpy.hstack((data, y)))