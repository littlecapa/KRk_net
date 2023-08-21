import random
import os
import logging

from torch.utils.data import Dataset, DataLoader
from utils.file_utils import save_checkpoint
from chess_utils.bit_board_interface import Bitboard_Interface

class Chess_Dataset(Dataset):

    def __init__(self, data_path):
        self.pos_fen = []
        self.mate = []
        logging.info(f"File: {data_path}")
        self.load_data(data_path)

    def __len__(self):
        return len(self.mate)

    def __getitem__(self, idx):
        #
        # To Do: Convert FEN to Bitvector!
        #
        return self.pos_fen[idx], self.mate[idx]
    
    def load_data(self, data_path):
        logging.debug(f"File: {data_path}")
        with open(data_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            values = line.strip().split(';')
            bbi = Bitboard_Interface(fen = values[0])
            self.pos_fen.append(bbi.get_13_64_bool_vector())  
            self.mate.append(abs(int(values[2])))
    
def fetch_dataloader(types, batch_size, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(params.base_dir, params.data_dir, split, params.data_file_name)

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(Chess_Dataset(path), batch_size=batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            else:
                dl = DataLoader(Chess_Dataset(path), batch_size=batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders
