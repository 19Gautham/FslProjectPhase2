
class Data(object):

    _instance = None

    training_data = None
    testing_data = None
    validation_data = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_training_data(self):
        return self.training_data

    def set_training_data(self, training_data):
        self.training_data = training_data

    def get_testing_data(self):
        return self.testing_data

    def set_testing_data(self, testing_data):
        self.testing_data = testing_data

    def get_validation_data(self):
        return self.validation_data

    def set_validation_data(self, validation_data):
        self.validation_data = validation_data
