import numpy as np

def normalize_data(data_array):
    #print(data_array.mean(axis=0))
    return ((data_array - data_array.mean(axis=0))/(data_array.std(axis=0)))

def calculate_training_error(output, expected_val):
    return np.square(expected_val - output)

def inverse_matrix():
    A = [[6, 6, 4], [6, 10, 5], [4, 5, 6]]
    y_transpose = [[1, 1, 1, -1, -1, -1], [1, 2, 2, 0, -1, 0], [1, 2, 0, 0, 0, -1]]
    margin_1 = [[1], [1], [1], [1], [1], [1]]
    margin_2 = [[1], [1], [1], [1], [1], [2]]
    print(np.linalg.det(A))
    matrix_inv = (np.dot(np.linalg.inv(A),np.linalg.det(A)))
    pseudo_inv = np.dot(matrix_inv, y_transpose)
    print(np.dot(pseudo_inv, margin_2))

if __name__ == "__main__":
    inverse_matrix()