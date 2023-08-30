"""Train the model"""

import argparse
import logging
import os

import torch
import torch.optim as optim

from utils.logger_utils import set_logger
from utils.param_utils import Params
from utils.file_utils import save_dict_to_json
from utils.stats_utils import save_stats
from utils.metrics_utils import metrics

import model.net as net
import data_loader as data_loader
from evaluate import evaluate
from trainer import Trainer
from loss import loss_fn, delta_loss_fn
from evaluate import evaluate

parser = argparse.ArgumentParser()

class Training_Manager():
    def __init__(self, loss_fn):
        self.trainer = Trainer(loss_fn = loss_fn)
    
    def start_training(self, metrics, params, restore):
        logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
        result = self.trainer.train_and_evaluate(metrics = metrics, params = params, restore = restore)
        return result

    def train_and_evaluate(self, metrics, params, restore):
        for i in range(params.len_param_combinations()):
            params.set_param_combination(i)
            dataloaders = data_loader.fetch_dataloader(
                    ['train', 'val', 'test'], batch_size = params.batch_size, params = params)
            train_dl = dataloaders['train']
            val_dl = dataloaders['val']
            test_dl = dataloaders['test']
            self.trainer.set_loader(train_dataloader = train_dl, val_dataloader = val_dl, test_dataloader = test_dl)
            model = net.KRk_Net_ChatGPT(params).cuda() if params.cuda else net.KRk_Net_ChatGPT(params)
            optimizer = optim.Adam(model.parameters(), params.learning_rate)
            self.trainer.set_optimizer(optimizer)
            self.trainer.set_model(model)
            results = self.start_training(metrics, params, restore)
            save_stats(params, results)
            
if __name__ == '__main__':

    # Load the parameters from json file
    parser.add_argument('--experiment_dir', default='experiments/base_model',
                    help="Directory containing params.json")
    args = parser.parse_args()
    json_path = os.path.join(args.experiment_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_data = None, json_path = json_path)
    params.set_opt_loss_info(optimizer = "Adam", loss_fn = loss_fn)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(params.seed)
    if params.cuda:
        torch.cuda.manual_seed(params.seed)

    # Set the logger
    set_logger(os.path.join(params.base_dir, params.log_dir, 'train.log'))

    manager = Training_Manager(loss_fn = torch.nn.CrossEntropyLoss())
    # Train the model
    logging.info("Starting training")
    manager.train_and_evaluate(metrics = metrics, params = params, restore = params.restore)
    logging.info("Ending training")