'''
Two bugs are fixed:
    1. padding side should be set to "right" for the decoder. If not, the attention mask would be messed up when overlapping the subsequent mask and the padding mask.
    2. a function that clamps the (maximum) value should be used before sending q@k.T into softmax:
        def clamp_pos_inf(t: torch.Tensor, clamp_max_only=True):
            # Clamp the tensor values
            if torch.isinf(t).any():
                clamp_value = torch.finfo(t.dtype).max - 100
                if clamp_max_only:
                    t = torch.clamp(t, max=clamp_value) # only clamp the positive infs
                else:
                    t = torch.clamp(t, max=clamp_value, min=-clamp_value) # only clamp the positive infs
            return t
        See /mntnfs/med_data5/guimingchen/anaconda3/envs/cgm/lib/python3.9/site-packages/torch/nn/functional.py, line ~4886.
'''

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)

import types
from models.overridden_classes_and_methods import (
    MultiheadAttention,
    overidden_decoder_forward,
    overridden_decoder_layer_forward,
    overridden_mha_block,
    overridden_sa_block,
)

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from tqdm import trange
from collections import OrderedDict

from transformers import AutoTokenizer
from torchtext.vocab import Vocab

import math
import pdb
from datetime import datetime


import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')



class Seq2SeqTransformer(pl.LightningModule):
    def __init__(self, hyperparams, num_encoder_layers: int, num_decoder_layers: int,
                 maxlen, emb_size: int, n_head: int, batch_first=True,
                 dim_feedforward:int = 512, weight_decay=4.5e-4, learning_rate=1e-4, dropout:float = 0.1,
                 src_tokenizer_name=None, tgt_tokenizer_name=None, do_not_override=False):
        super(Seq2SeqTransformer, self).__init__()
        self.save_hyperparameters(hyperparams)
        self.hyperparameters = hyperparams
        # tokenizer
        self.src_vocab = torch.load('data/vocab_english.pt', map_location='cpu')
        self.tgt_vocab = torch.load('data/vocab_chinese.pt', map_location='cpu')
        self.src_tokenizer = Tokenizer(src_tokenizer_name, padding_side='right', maxlen=maxlen)
        self.tgt_tokenizer = Tokenizer(tgt_tokenizer_name, padding_side='right', maxlen=maxlen)

        
        self.transformer = nn.Transformer(
            d_model=emb_size, 
            nhead=n_head, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
            norm_first=True
        )
        self.do_not_override = do_not_override
        if not do_not_override:
            # TODO: override self.transformer.decoder.forward() as overidden_decoder_forward()
            # TODO: override mod._sa_block() as overridden_sa_block()
            # TODO: override mod._mha_block() as overridden_mha_block()
            # TODO: override mod.forward() as overridden_decoder_layer_forward()
            # TODO: check that this does not break anything during training
            logging.info('using overriden methods. Useful at inference.')
            self.transformer.decoder.forward = types.MethodType(overidden_decoder_forward, self.transformer.decoder)
            for layer in self.transformer.decoder.layers:
                layer.self_attn = MultiheadAttention(emb_size, n_head, dropout=dropout, batch_first=batch_first,)
                layer.multihead_attn = MultiheadAttention(emb_size, n_head, dropout=dropout, batch_first=batch_first)
                layer._sa_block = types.MethodType(overridden_sa_block, layer)
                # factory_kwargs = {'device': device, 'dtype': dtype}
                layer._mha_block = types.MethodType(overridden_mha_block, layer)
                layer.forward = types.MethodType(overridden_decoder_layer_forward, layer)
        
        self.classifier = nn.Linear(emb_size, self.tgt_tokenizer.vocab_size)

        self.src_tok_emb = TokenEmbedding(self.src_tokenizer.vocab_size, emb_size, self.src_tokenizer.pad_token_id)
        self.tgt_tok_emb = TokenEmbedding(self.tgt_tokenizer.vocab_size, emb_size, self.tgt_tokenizer.pad_token_id)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout, maxlen=maxlen)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_size = emb_size
        self.maxlen = maxlen

        # for optimizer
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate

        # criterion
        # Using normal CE loss leads to a non-decreasing loss at training. To be figured out why...
        # self.criterion = nn.CrossEntropyLoss(ignore_index=self.tgt_tokenizer.pad_token_id, label_smoothing=0.1)
        self.criterion = TranslationLoss(self.tgt_tokenizer.pad_token_id)

    

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,):
        token_emb = self.src_tok_emb(src)
        pos_emb = self.positional_encoding(src)
        src_emb = self.emb_dropout(token_emb + pos_emb)

        token_emb = self.tgt_tok_emb(tgt)
        pos_emb = self.positional_encoding(tgt)
        tgt_emb = self.emb_dropout(token_emb + pos_emb)
        
        tgt_mask, src_key_padding_mask, tgt_key_padding_mask = self.create_mask(src, tgt)

        if self.do_not_override:
            outs = self.transformer(src_emb, tgt_emb, 
                                    tgt_mask=tgt_mask, 
                                    src_key_padding_mask=src_key_padding_mask, 
                                    tgt_key_padding_mask=tgt_key_padding_mask)
        else:
            outs, _ = self.transformer(src_emb, tgt_emb, 
                                    tgt_mask=tgt_mask, 
                                    src_key_padding_mask=src_key_padding_mask, 
                                    tgt_key_padding_mask=tgt_key_padding_mask)
        
        return outs
    

    @torch.no_grad()
    def forward_with_past(self, src: torch.Tensor, tgt: torch.Tensor, 
                          tgt_mask, src_key_padding_mask, tgt_key_padding_mask,
                          cache: dict={}):

        # get encoder_output
        if cache.get('encoder_output', None) is not None:
            encoder_output = cache.get('encoder_output', None)
        else:
            # src_emb: bs, fixed_len, dim
            token_emb = self.src_tok_emb(src)
            pos_emb = self.positional_encoding(src)
            src_emb = self.emb_dropout(token_emb + pos_emb)

            encoder_output = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
            cache['encoder_output'] = encoder_output
        
        past = cache.get('past', None)
        past_length = cache.get('past_length', None)
        if past is not None:
            assert past_length > 0
            past = torch.cat(past, dim=-2)  # (n_layer, 2, bs*nh, len_past, dim_head)
            pos_tensor = torch.ones(tgt.shape[0], past_length+1).long()
            pos_emb = self.positional_encoding(pos_tensor)[:, past_length, :].unsqueeze(-2)
        else:
            assert past_length == 0
            pos_tensor = torch.ones(tgt.shape[0], past_length+1).long()
            pos_emb = self.positional_encoding(pos_tensor)[:, past_length, :].unsqueeze(-2)
        token_emb = self.tgt_tok_emb(tgt)
        tgt_emb = self.emb_dropout(token_emb + pos_emb)
        
        # tgt_emb: bs, growing_len, dim
        # encoder_output: bs, fixed_len, dim

        outs, present = self.transformer.decoder(tgt_emb, encoder_output,
                                    tgt_mask=tgt_mask, memory_mask=None,
                                    memory_key_padding_mask=src_key_padding_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    past=past, return_present=True)
        
        # update past
        if past is None:
            cache['past'] = [present]
        else:
            cache['past'].append(present)

        return outs, cache
    
    def shared_step(self, batch, batch_idx):
        src, tgt = batch['src'], batch['trg']
        self.batch_size = len(src)
        src, _ = self.src_tokenizer(src, padding=True, truncation=True, return_tensors="pt")
        tgt, n_tokens = self.tgt_tokenizer(tgt, padding=True, truncation=True, return_tensors="pt")

        src, tgt = map(lambda x: x.to(self.device), (src, tgt)) # (bs, seqlen)

        tgt_input = tgt[:, :-1]

        outs = self.forward(src, tgt_input)
        logits = self.classifier(outs)

        tgt_y = tgt[:,1:]
        loss = self.criterion(
            logits.contiguous().view(-1, logits.size(-1)), tgt_y.contiguous().view(-1)
        ) / (n_tokens-len(tgt_y)) # normalized by the number of non-padding tokens in tgt_y

        if torch.isnan(loss):
            pdb.set_trace()

        return loss
    

    @torch.no_grad()
    def generate(self, src, do_sample=True, temperature=1.0, use_cache=True, verbose=False):

        if isinstance(src, str):
            src = [src]
        
        tgt = torch.tensor([self.tgt_tokenizer.bos_token_id]*len(src))[:, None] # (bs, 1)
        src, _ = self.src_tokenizer(src, padding=True, truncation=True, return_tensors="pt")
        src, tgt = map(lambda x: x.to(self.device), [src, tgt])
        
        sample = tgt
        cond_len = tgt.shape[1]
        
        cache = {} if use_cache else None
        if verbose:
            print('start generating')

        for cur_len in range(self.maxlen):
            # this is the index of current tgt's position emb
            if use_cache:
                cache['past_length'] = cond_len + cur_len -1 

            tgt_mask, src_key_padding_mask, tgt_key_padding_mask = self.create_mask(src, sample)


            if use_cache:
                out, cache = self.forward_with_past(src, tgt, 
                                                    tgt_mask, src_key_padding_mask, tgt_key_padding_mask,
                                                    cache)
                
                logits = self.classifier(out[:, -1]) # (bs, vocab)

                logits /= temperature
                # mask pad_token
                logits[..., self.tgt_tokenizer.pad_token_id] = -torch.inf

                if do_sample:
                    prob = F.softmax(logits, dim=-1)
                    tgt = torch.multinomial(prob, num_samples=1)# (bs, 1)
                else:
                    tgt = torch.argmax(logits, dim=1, keepdim=True) # (bs, 1)
            
            else:
                out = self.forward(src, sample)
                logits = self.classifier(out[:, -1]) # (bs, vocab)

                logits /= temperature
                logits[..., self.tgt_tokenizer.pad_token_id] = -torch.inf

                if do_sample:
                    prob = F.softmax(logits, dim=-1)
                    tgt = torch.multinomial(prob, num_samples=1)# (bs, 1)
                else:
                    tgt = torch.argmax(logits, dim=1, keepdim=True) # (bs, 1)
                
                # tgt = torch.concat([tgt, y], dim=1)

            sample = torch.concat([sample, tgt], dim=1)


        tgt = self.tgt_tokenizer.batch_decode(sample, self.tgt_vocab)
        return tgt # List[str]

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
        self.log('lr', self.lr_schedulers().get_last_lr()[0], prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
        if self.global_step % self.trainer.print_every == 0 and self.global_rank == 0:
            logging.info(f'step: {self.global_step}, train_loss: {loss.item():.4f}, lr: {self.lr_schedulers().get_last_lr()[0]:e} @ {datetime.now()}')

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        
        self.log("val/loss", loss, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
        if self.global_step % self.trainer.print_every == 0 and self.global_rank == 0:
            logging.info(f'step: {self.global_step}, val_loss: {loss.item():.4f}, lr: {self.lr_schedulers().get_last_lr()[0]:e} @ {datetime.now()}')
        return loss
    
    def create_mask(self, src, tgt,):
        '''
        src: (bs, seqlen)
        tgt: (bs, seqlen)
        '''
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to(self.device)
        src_key_padding_mask = (src == self.src_tokenizer.pad_token_id)
        tgt_key_padding_mask = (tgt == self.tgt_tokenizer.pad_token_id)
        return tgt_mask.to(self.device), src_key_padding_mask.to(self.device), tgt_key_padding_mask.to(self.device)


    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optim_groups = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = torch.optim.Adam(optim_groups, **self.hyperparameters.optimizer.general)
        scheduler = MyScheduler(optimizer, dim_embed=self.emb_size, 
                                warmup_steps=self.hyperparameters.scheduler.warmup_steps,
                                verbose=self.hyperparameters.scheduler.verbose)
        # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=1, threshold=0.5)
        returned = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hyperparameters.scheduler.monitor,
                    "frequency": self.hyperparameters.scheduler.frequency,
                    "interval": self.hyperparameters.scheduler.interval,
                },
            }
        # self.hyperparams.update(returned)
        
        return returned
    


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_ids: torch.LongTensor):
        '''
        token_ids: (bs, seqlen)
        pos_embedding: (maxlen, 1, dim)
        '''

        pos_embedding = self.pos_embedding.transpose(0,1)
        return pos_embedding[:, :token_ids.shape[1],:]

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, padding_idx):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.emb_size = emb_size
    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long())



from torch.optim.lr_scheduler import _LRScheduler

class MyScheduler(_LRScheduler):
    # https://kikaben.com/transformers-training-details/
    def __init__(self, 
                 optimizer,
                 dim_embed: int,
                 warmup_steps: int,
                 last_epoch: int=-1,
                 verbose: bool=False) -> None:

        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self) -> float:
        lr = calc_lr(self._step_count, self.dim_embed, self.warmup_steps)
        return [lr] * self.num_param_groups


def calc_lr(step, dim_embed, warmup_steps):
    return dim_embed**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))


class TranslationLoss(nn.Module):
    
    def __init__(self, padding_idx):
        super(TranslationLoss, self).__init__()
        # 
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx

    def forward(self, inp, target):
        """
        :param input: logits (bs*seqlen, vocab_size)
        :param target: labels (bs*seqlen)
        :return: loss
        """

        """
        KLDivLoss requires a logsoftmaxed input
        """
        inp = F.log_softmax(inp, dim=-1)

        """
        construct label distribution, 
        converting [[1, 34, 15, ...]] into
        [[[0, 1, 0, ..., 0],
          [0, ..., 1, ..,0],
          ...]],
        ...]
        """
        # create a tensor of all zeros
        true_dist = torch.zeros(inp.size()).to(inp.device)
        # fill the position corresponding to the target label with 1, along dim=1 
        true_dist.scatter_(1, target.data.unsqueeze(1), 1)
        # mask pad tokens from being calculated in loss
        mask = torch.nonzero(target.data == self.padding_idx) # (bs*seqlen, 1), the index of nonzero entries
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)

        # calculate loss. Note that we use reduction='sum'
        # return self.criterion(inp, true_dist.clone().detach())
        return self.criterion(inp, true_dist)
    

class Tokenizer():
    def __init__(self, name, padding_side='right', maxlen=256,) -> None:
        if name == 'bert-base-uncased':
            self.lang = 'english'
            self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        elif name == 'bert-base-chinese':
            self.lang = 'chinese'

        self.padding_side = padding_side
        self.maxlen = maxlen
        self.init_vocab()
        self.init_special_tokens()
    
    def init_vocab(self):
        print('init vocab {}'.format(self.lang))
        self.vocab: Vocab = torch.load(f'data/vocab_{self.lang}.pt', map_location="cpu")
        self.vocab_size = len(self.vocab.vocab)

    def init_special_tokens(self):
        self.bos_token, self.eos_token, self.pad_token, self.unk_token\
            = ["<s>", "</s>", "<pad>", "<unk>"]
        self.bos_token_id, self.eos_token_id, self.pad_token_id, self.unk_token_id\
              = self.vocab.lookup_indices(["<s>", "</s>", "<pad>", "<unk>"])

    def _pre_tokenize(self, line: str):
        if self.lang == 'english':
            encoded = self.bert_tokenizer.encode(line, add_special_tokens=False)
            encoded = self.bert_tokenizer.batch_decode([[i] for i in encoded])

        elif self.lang == 'chinese':
            encoded = list(line.strip().replace(" ", ""))
            
        return encoded

    def __call__(self, sequences: list[str], padding=True, truncation=True, return_tensors='pt'):
        # ["<s>", "</s>", "<pad>", "<unk>"] -> [0,1,2,3]
        '''
        sequences: ['i love deep learning', 'attention is powerful', ...]
        '''
        encoded = []
        for s in sequences:
            line = self._pre_tokenize(s) # ['i', 'love', 'deep', 'learning']
            line = self.vocab(line) # [5, 99, 102, 78]
            line = [self.bos_token_id] + line + [self.eos_token_id] # add bos and eos: [0,5,99,102,78,1]
            encoded.append(line)
        # add bos and eos
        # encoded = list(map(lambda li: [self.bos_token_id] + li + [self.eos_token_id], encoded))
        batch_max_len = max(map(len, encoded))
        batch_max_len = min(self.maxlen, batch_max_len)
        # calculate n_tokens for normalizing KLDivn loss
        n_tokens = sum(map(len, encoded))

        if padding == True:
            # pad to batch_max_len
            for i in range(len(encoded)):
                new = encoded[i] + [self.pad_token_id] * max(0, batch_max_len-len(encoded[i]))
                encoded[i] = new 
        else:
            raise NotImplementedError
        
        if truncation == True:
            encoded = list(map(lambda li: li[:self.maxlen], encoded))
        
        if return_tensors == 'pt':
            return torch.tensor(encoded), n_tokens # batched tokens
        

        
    
        
    def batch_decode(self, sequences, zh_vocab: Vocab):
        '''
        batch: bs, seqlen
        '''
        sequences = sequences.detach().cpu().numpy().tolist()
        decoded = []
        for seq in sequences:
            try:
                eos_pos = seq.index(self.eos_token_id) # locate the first eos token
            except ValueError:
                eos_pos = len(seq)
            seq = seq[1:eos_pos] # fetch all tokens bwtween bos and eos
            seq_d = zh_vocab.lookup_tokens(seq) # list[str]
            seq_d = ''.join(seq_d)
            decoded.append(seq_d)

        return decoded