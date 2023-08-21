import json
import copy
import logging

class DictWithAttributes(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            raise AttributeError(f"'DictWithAttributes' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

class Params():
    """Class that loads all parameters from a json file."""

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__
    
    def get_training_params_dict(self, learning_rate, batch_size, num_epochs, dropout_rate):
        deep_copied_dict = DictWithAttributes(copy.deepcopy(self.__dict__))
        print(f"DeepCopy: {deep_copied_dict}")
        deep_copied_dict.learning_rate = learning_rate
        deep_copied_dict.batch_size = batch_size
        deep_copied_dict.num_epochs = num_epochs
        deep_copied_dict.dropout_rate = dropout_rate
        return deep_copied_dict
    
def get_training_params_csv_str(training_params):
    output = str(training_params.learning_rate) +";"
    output += str(training_params.batch_size) +";"
    output += str(training_params.num_epochs) +";"
    output += str(training_params.dropout_rate)
    return output