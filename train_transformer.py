import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin


from utils.transformer_utils import load_config, load_model, load_nmt_loader, get_args, parse_args

import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
torch.manual_seed(1)



def main():
    config = load_config()
    arg = get_args()
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
        monitor='val/loss',
        dirpath=f'output/{logger.name}/version_{logger.version}',
        filename='epoch{epoch:02d}-val_loss{val/loss:.2f}',
        auto_insert_metric_name=False,
        every_n_epochs=1,
        save_weights_only=False, save_last=True, save_top_k=1,
    )
    trainer = pl.Trainer(**config.trainer, callbacks=[checkpoint_callback], logger=logger, 
                         plugins=DDPPlugin(find_unused_parameters=False),
                         gradient_clip_val=0.5)
    trainer.print_every = 50

    print('*'*50)
    print(f'this is {logger.version}')
    print('*'*50)

    model = load_model(config) # hyperparams saved
    trainer.fit(model, train_loader, val_loader, )

    print(f'version {logger.version} done.')





if __name__ == '__main__':
    import sys
    main()
'''
TOKENIZERS_PARALLELISM=true python train_transformer.py --gpus 1 --name debug --version debug --train_batch_size 10
'''