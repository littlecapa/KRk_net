import json
import logging
import os
import shutil
from itertools import product
import torch

class Params():
    """Class that loads hyperparameters from a json file.
    """

    def __init__(self, json_data = None, json_path = "params.json"):
      if json_data is None:
        with open(json_path) as f:
            params = json.load(f)
      else:
        params = json_data
      self.__dict__.update(params)
      # Check for Lists
      # Iterate through the dictionary
      self.key_list = []
      self.value_list = []
      for key, value in params.items():
        if key.endswith("_list"):
          # Create a new key without "_list"
          new_key = key[:-5]  # Remove "_list" from the end of the key
          self.key_list.append(new_key)
          self.value_list.append(value)
      self.combinations = list(enumerate(product(*self.value_list)))
    
    def len_param_combinations(self):
      return len(self.combinations)

    def set_param_combination(self, comb_index):
      for key_index, key in enumerate(self.key_list):
        self.add_param(key, self.combinations[comb_index][1][key_index])

    def add_param(self, key, value):
      self.__dict__[key] = value

    def set_opt_loss_info(self,optimizer, loss_fn, info = ""):
      self.add_param("optimizer", optimizer)
      self.add_param("loss_fn", loss_fn)
      self.add_param("info", info)

    def get_param_value(self, key, safe = False):
      if key in self.__dict__:
          return self.__dict__[key]
      else:
        if safe:
          raise KeyError(f"Parameter '{key}' not found in Parameters.")
        else:
          return ""

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

    stat_params = ["learning_rate", "batch_size", "num_epochs", "dropout_rate", "optimizer", "loss_fn", "info"]

    def get_stats_str(self):
      out = ""
      for p in self.stat_params:
        out += f"{self.get_param_value(p)};"
      return out