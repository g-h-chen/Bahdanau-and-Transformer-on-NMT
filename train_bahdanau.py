import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
import os
import json
from omegaconf import OmegaConf

from transformers import AutoTokenizer
from argparse import ArgumentParser

from models.bahdanau import EncoderDecoder

import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
torch.manual_seed(0)

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--name', default=None, type=str)
    parser.add_argument('--version', default=None)
    parser.add_argument('--n_enc', type=int, default=None)
    parser.add_argument('--n_dec', type=int, default=None)
    parser.add_argument('--bs_train', type=int, default=None)
    parser.add_argument('--bs_val', type=int, default=None)
    parser.add_argument('--train_length', type=int, default=None)
    parser.add_argument('--val_length', type=int, default=None)
    parser.add_argument('--limit_val_batches', type=int, default=None)
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--hidden_size', type=int, default=None)
    parser.add_argument('--enable_progress_bar', action='store_true', default=None)
    parser.add_argument('--warmup_steps', type=int, default=None)
    parser.add_argument('--val_check_interval', type=int, default=None)
    parser.add_argument('--tf_code', type=int, default=None)
    # parser.add_argument('--teacher_or_self',  action='store_true')
    parser.add_argument('--bidirectional',  action='store_true')
    # parser.add_argument('--')
    args = parser.parse_args()

    return args

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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=48) # bg
    return dataloader


def load_config():
    config_pth = 'utils/bahdanau_config.yaml'
    config = OmegaConf.load(config_pth)
    return config

def parse_args(config, arg):
    if arg.version is not None:
        arg.version = f'version_{arg.version}'
        config.logger.version = arg.version

    if arg.name is not None:
        config.logger.name = arg.name
    
    if arg.bs_train:
        config.data.train_batch_size = arg.bs_train
    if arg.train_length:
        config.data.train_length = arg.train_length
    if arg.bs_val:
        config.data.val_batch_size = arg.bs_val
    if arg.val_length:
        config.data.val_length = arg.val_length
        
    if arg.n_enc is not None:
        config.EncoderDecoder.num_encoder_layers = arg.n_enc
    if arg.n_dec is not None:
        config.EncoderDecoder.num_decoder_layers = arg.n_dec
    if arg.hidden_size is not None:
        config.EncoderDecoder.hidden_size = arg.hidden_size

    if arg.gpus is not None:
        config.trainer.gpus = arg.gpus
    if arg.tf_code is not None:
        config.tf.code = arg.tf_code

    if arg.enable_progress_bar is not None:
        config.trainer.enable_progress_bar = arg.enable_progress_bar
    if arg.limit_val_batches is not None:
        config.trainer.limit_val_batches = arg.limit_val_batches
    if arg.val_check_interval is not None:
        config.trainer.val_check_interval = arg.val_check_interval
    if arg.warmup_steps is not None:
        config.scheduler.warmup_steps = arg.warmup_steps

    # if arg.teacher_or_self :
    #     config.EncoderDecoder.teacher_or_self = True
    if arg.bidirectional :
        config.EncoderDecoder.bidirectional = True

    return config

def load_model(config):    
    logging.info('loading model')
    model = EncoderDecoder(config, **config.EncoderDecoder,)
    return model

def main():
    arg = get_args()

    config = load_config()
    config = parse_args(config, arg)

    train_loader, val_loader = \
        load_nmt_loader(split='train', batch_size=config.data.train_batch_size, length=config.data.train_length), \
            load_nmt_loader(split='val', batch_size=config.data.val_batch_size, length=config.data.val_length),
    
    config.data.train_length = len(train_loader.dataset)
    config.data.val_length = len(val_loader.dataset)



    logger = TensorBoardLogger(**config.logger)
    if type(logger.version) == int:
        config.logger.version = f'version_{logger.version}'

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'output/{logger.name}/{config.logger.version}',
        **config.ModelCheckpoint,
    )
    trainer = pl.Trainer(**config.trainer, callbacks=[checkpoint_callback], logger=logger, 
                         plugins=DDPPlugin(find_unused_parameters=False),
                         )

    trainer.print_every = 100

    print('*'*50)
    print(f'this is {config.logger.version}')
    print('*'*50)

    model = load_model(config) # hyperparams saved
    trainer.fit(model, train_loader, val_loader, )

    print(f'{config.logger.version} done.')





if __name__ == '__main__':
    import sys
    main()
'''
TOKENIZERS_PARALLELISM=true python train_bahdanau.py --name debug --version debug --n_enc 2 --n_dec 2 
'''