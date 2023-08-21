import json
import logging
import os
import shutil
from datetime import datetime
import torch

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

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


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)
        
BEST_DICT_JSON = "metrics_val_best_weights.json"
DEFAULT_DICT_JSON = "metrics_val_last_weights.json"
DEFAULT_CHECKPOINT_FILENAME = "last.pth.tar"
BEST_CHECKPOINT_FILENAME = "best.pth.tar"

def save_dict_to_json(d, base_dir, checkpoint_dir, json_file_name = DEFAULT_DICT_JSON):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    json_path = os.path.join(base_dir, checkpoint_dir, json_file_name)
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

def save_checkpoint(state, is_best, base_dir, checkpoint_dir):

    filepath = os.path.join(base_dir, checkpoint_dir, DEFAULT_CHECKPOINT_FILENAME)
    if not os.path.exists(os.path.join(base_dir, checkpoint_dir)):
        print("Checkpoint Directory does not exist! Making directory {}".format(os.path.join(base_dir, checkpoint_dir)))
        os.mkdir(os.path.join(base_dir, checkpoint_dir))
    else:
        print("Checkpoint Directory exists! {}".format(checkpoint_dir))
    torch.save(state, filepath)
    print("Checkpoint saved! {}".format(filepath))
    if is_best:
        shutil.copyfile(filepath, os.path.join(base_dir, checkpoint_dir, BEST_CHECKPOINT_FILENAME))


def load_checkpoint(model, base_dir, checkpoint_dir, optimizer=None, restore_file = DEFAULT_CHECKPOINT_FILENAME):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    checkpoint = os.path.join(base_dir, checkpoint_dir, restore_file)
    if not os.path.exists(checkpoint_dir):
        raise("File doesn't exist {}".format(checkpoint_dir))
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

def write_line_to_csv_file(base_dir, stats_dir, filename, line):
    # Get the current timestamp as a datetime object
    current_time = datetime.now()
    # Convert the datetime object to a string in a specific format
    timestamp_string = current_time.strftime("%d-%m-%Y %H:%M:%S")
    filename = os.path.join(base_dir, stats_dir, filename)
    if not os.path.exists(filename):
        raise("File doesn't exist {}".format(filename))
    with open(filename, 'a') as f:
        f.write(f"{timestamp_string};{line}\n")