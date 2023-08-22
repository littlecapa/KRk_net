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
        self.trainer = Trainer(loss_fn = delta_loss_fn)
    
    def start_training(self, metrics, training_params, restore):
        logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
        result = self.trainer.train_and_evaluate(metrics = metrics, params = training_params, restore = restore)
        return result

    def train_and_evaluate(self, metrics, params, restore):
        for dropout_rate in params.dropout_rate:
            for batch_size in params.batch_size:
                # fetch dataloaders
                dataloaders = data_loader.fetch_dataloader(
                    ['train', 'val', 'test'], batch_size = batch_size, params = params)
                train_dl = dataloaders['train']
                val_dl = dataloaders['val']
                test_dl = dataloaders['test']
                self.trainer.set_loader(train_dataloader = train_dl, val_dataloader = val_dl, test_dataloader = test_dl)
                for learning_rate in params.learning_rate:                  
                    for num_epochs in params.num_epochs:
                        model = net.KRk_Net_ChatGPT(dropout_rate, params).cuda() if params.cuda else net.KRk_Net_ChatGPT(dropout_rate, params)
                        optimizer = optim.Adam(model.parameters(), learning_rate)
                        self.trainer.set_optimizer(optimizer)
                        self.trainer.set_model(model)
                        training_params = params.get_training_params_dict(learning_rate, batch_size, num_epochs, dropout_rate)
                        results, summary = self.start_training(metrics, training_params, restore)
                        save_stats(training_params, results, summary, optimizer = "Adam", loss_fn = "loss.py")

if __name__ == '__main__':

    # Load the parameters from json file
    parser.add_argument('--experiment_dir', default='experiments/base_model',
                    help="Directory containing params.json")
    args = parser.parse_args()
    json_path = os.path.join(args.experiment_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(params.seed)
    if params.cuda:
        torch.cuda.manual_seed(params.seed)

    # Set the logger
    set_logger(os.path.join(params.base_dir, params.log_dir, 'train.log'))

    manager = Training_Manager(loss_fn = delta_loss_fn)
    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    manager.train_and_evaluate(metrics = metrics, params = params, restore = params.restore)
    logging.info("Ending training for {} epoch(s)".format(params.num_epochs))