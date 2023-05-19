from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import json
from omegaconf import OmegaConf

import os
from models.transformer import Seq2SeqTransformer

import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
torch.manual_seed(1)

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--name', default=None, type=str)
    parser.add_argument('--version', default=None)
    parser.add_argument('--n_enc', type=int, default=None)
    parser.add_argument('--n_dec', type=int, default=None)
    parser.add_argument('--train_batch_size', type=int, default=None)
    parser.add_argument('--val_batch_size', type=int, default=None)
    parser.add_argument('--train_length', type=int, default=None)
    parser.add_argument('--val_length', type=int, default=None)
    parser.add_argument('--limit_val_batches', type=int, default=None)
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--hidden_size', type=int, default=None)
    parser.add_argument('--enable_progress_bar', action='store_true', default=None)
    parser.add_argument('--warmup_steps', type=int, default=None)
    parser.add_argument('--val_check_interval', type=int, default=None)
    parser.add_argument('--norm_first',  action='store_true')
    parser.add_argument('--do_not_override',  action='store_true')
    # parser.add_argument('--')
    args = parser.parse_args()

    return args

def parse_args(config, arg):
    assert arg.name and arg.version, 'must specify the experiment name and version.'
    if arg.name is not None:
        config.logger.name = arg.name

    if arg.version is not None:
        arg.version = f'version_{arg.version}'
        config.logger.version = arg.version

    
    if arg.train_batch_size:
        config.data.train_batch_size = arg.train_batch_size
    if arg.train_length:
        config.data.train_length = arg.train_length
    if arg.val_batch_size:
        config.data.val_batch_size = arg.val_batch_size
    if arg.val_length:
        config.data.val_length = arg.val_length
        
    if arg.n_enc is not None:
        config.seq2seqtransformer.num_encoder_layers = arg.n_enc
    if arg.n_dec is not None:
        config.seq2seqtransformer.num_decoder_layers = arg.n_dec
    if arg.hidden_size is not None:
        config.seq2seqtransformer.hidden_size = arg.hidden_size
    if arg.norm_first:
        config.seq2seqtransformer.norm_first = arg.norm_first
    if arg.do_not_override:
        config.seq2seqtransformer.do_not_override = arg.do_not_override

    if arg.gpus is not None:
        config.trainer.gpus = arg.gpus

    if arg.enable_progress_bar is not None:
        config.trainer.enable_progress_bar = arg.enable_progress_bar
    if arg.limit_val_batches is not None:
        config.trainer.limit_val_batches = arg.limit_val_batches
    if arg.val_check_interval is not None:
        config.trainer.val_check_interval = arg.val_check_interval
    if arg.warmup_steps is not None:
        config.scheduler.warmup_steps = arg.warmup_steps

    return config



class NMTDataset():
    def __init__(self, split="train", length=2000):
        super().__init__()
        self.src = 'english'
        self.trg = 'chinese'
        self.split = split

        if split == 'train':
            self.length = min(length, 5161434)
        elif split == 'val':
            self.length = min(length, 39323)
        print(f'{self.split} length: {self.length}')

        root = "data"
        pth = os.path.join(root, f'{split}.jsonl')
        with open(pth) as f:
            self.texts = [json.loads(line) for line in f][:self.length] 

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        src_text = self.texts[idx][self.src]
        trg_text = self.texts[idx][self.trg]

        return {'src': src_text, 'trg': trg_text}

        

def load_nmt_loader(split='train', batch_size=64, length=1000):
    dataset = NMTDataset(split, length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'), num_workers=48)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1) # dbg
    return dataloader


def load_config():
    config_pth = 'utils/transformer_config.yaml'
    config = OmegaConf.load(config_pth)
    return config


def load_model(config):    
    logging.info('loading model')
    model = Seq2SeqTransformer(config, **config.seq2seqtransformer,)
    return model
