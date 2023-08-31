"""Train the model"""

import logging
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import torch
import io

import model.net as net
import data_loader as data_loader
from utils.stats_utils import RunningAverage, Summary
from utils.file_utils import load_checkpoint, save_checkpoint, save_dict_to_json, BEST_DICT_JSON, BEST_CHECKPOINT_FILENAME

class Trainer():

    def __init__(self, loss_fn):
        self.loss_fn = loss_fn
        self.qa_list = []

    def __del__(self):
        if len(self.qa_list) > 0:
            with io.open("qa.csv", 'a') as f:
                for qa_result in self.qa_list:
                    expected, actual = qa_result
                    f.write(f"{expected};{actual}\n")

    def add_qa_result(self, expected, actual):
        max_index = torch.argmax(actual, dim=-1)
        new_qa = (expected.item(), max_index.item())
        self.qa_list.append(new_qa)

    def add_qa_result_tensors(self, labels_batch, output_batch):
        batch_size = output_batch.size()[0]
        for i in range(batch_size):
            self.add_qa_result(labels_batch[i], output_batch[i])


    def set_model(self, model):
        self.model = model
    
    def set_loader(self, train_dataloader, val_dataloader, test_dataloader):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def run_model(self, loss_fn, loader, metrics, params, training = False, qa = False):
        if training:
            self.model.train()
        else:
            self.model.eval()

        summ = []
        loss_avg = RunningAverage()

        # Use tqdm for progress bar
        with tqdm(total=len(loader)) as t:
            for i, (input_batch, labels_batch) in enumerate(loader):
                # move to GPU if available
                if params.cuda:
                    input_batch, labels_batch = input_batch.cuda(
                        non_blocking=True), labels_batch.cuda(non_blocking=True)
                # convert to torch Variables
                # According to ChatGPT not necessary any more
                # input_batch, labels_batch = Variable(input_batch), Variable(labels_batch)

                # compute model output and loss
                output_batch = self.model(input_batch)
                if qa:
                    self.add_qa_result_tensors(labels_batch, output_batch)
                loss = loss_fn(output_batch, labels_batch)
        
                if training:
                    # clear previous gradients, compute gradients of all variables wrt loss
                    self.optimizer.zero_grad()
                    loss.backward()

                    # performs updates using calculated gradients
                    self.optimizer.step()

                # Evaluate summaries only once in a while
                if (i % params.save_summary_steps == 0) or not training:
                    # extract data from torch Variable, move to cpu, convert to numpy arrays
                    output_batch = output_batch.data.cpu().numpy()
                    labels_batch = labels_batch.data.cpu().numpy()
                    # compute all metrics on this batch
                    summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                    for metric in metrics}
                    summary_batch['loss'] = loss.item()
                    summ.append(summary_batch)

                if training:
                    # update the average loss
                    loss_avg.update(loss.item())

                    t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
                    t.update()

        # compute mean of all metrics in summary
        metrics_mean = {metric: np.mean([x[metric]
                                        for x in summ]) for metric in summ[0]}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                    for k, v in metrics_mean.items())
        logging.info(f"!- Train metrics: {metrics_string} {len(summ[0])}")
        return metrics_mean, summary_batch
    
    def qa(self, metrics, params):
        load_checkpoint(model = self.model, base_dir = params.base_dir, checkpoint_dir = params.checkpoint_dir, optimizer = None, restore_file =  BEST_CHECKPOINT_FILENAME)
        for loader in [self.test_dataloader, self.val_dataloader]:
            _, _ = self.run_model(loss_fn = self.loss_fn, loader = loader, metrics = metrics, params = params, training = False, qa = True)
        

    def train_and_evaluate(self, metrics, params, restore):
        """Train the model and evaluate every epoch.

        Args:
            model: (torch.nn.Module) the neural network
            train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
            val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
            optimizer: (torch.optim) optimizer for parameters of model
            loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
            metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
            params: (Params) hyperparameters
            restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
        """
        # reload weights from restore_file if specified
        if restore:
            logging.info("Restoring parameters from ")
            load_checkpoint(model = self.model, base_dir = params.base_dir, checkpoint_dir = params.checkpoint_dir, optimizer = self.optimizer)
        best_val_acc = 0.0
        summary_collector = Summary(params)

        for epoch in range(params.num_epochs):
            # Run one epoch
            logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

            logging.info("Training")
            # compute number of batches in one epoch (one full pass over the training set)
            _, summary = self.run_model(loss_fn = self.loss_fn, loader = self.train_dataloader, metrics = metrics, params = params, training = True)
            summary_collector.add(summary, "Training")

            # Evaluate for one epoch on validation set
            logging.info("Validation")
            val_metrics, summary = self.run_model(loss_fn = self.loss_fn, loader = self.val_dataloader, metrics = metrics, params = params, training = False)
            summary_collector.add(summary, "Validation")

            val_acc = val_metrics['accuracy']
            is_best = val_acc >= best_val_acc

            # Save weights
            save_checkpoint(state = {'epoch': epoch + 1,
                                'state_dict': self.model.state_dict(),
                                'optim_dict': self.optimizer.state_dict()},
                                is_best=is_best,
                                base_dir = params.base_dir,
                                checkpoint_dir = params.checkpoint_dir)

            # If best_eval, best_save_path
            if is_best:
                logging.info("- Found new best accuracy")
                best_val_acc = val_acc

                # Save best val metrics in a json file in the model directory
                save_dict_to_json(d = val_metrics, base_dir = params.base_dir, checkpoint_dir = params.checkpoint_dir, json_file_name = BEST_DICT_JSON)
                
            # Save latest val metrics in a json file in the model directory
            save_dict_to_json(d = val_metrics, base_dir = params.base_dir, checkpoint_dir = params.checkpoint_dir)

        #
        # Final Evaluation Test
        #
        logging.info("Testing")
        test_metrics, summary = self.run_model(loss_fn = self.loss_fn, loader = self.test_dataloader, metrics = metrics, params = params, training = False)
        summary_collector.add(summary, "Testing")
        logging.info(f"Test Metrics: {test_metrics}")
        summary_collector.save()
        return test_metrics