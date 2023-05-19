

import random
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


from transformers import AutoTokenizer
from torchtext.vocab import Vocab

import math
import pdb
from datetime import datetime

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')





class EncoderRNN(nn.Module):
    def __init__(self, input_vocab_size, hidden_size, num_layers, dropout, bidirectional=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_vocab_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout, 
            bidirectional=bidirectional,
        )

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded)
        return output, hidden # (bs, seqlen, dim), (D*n_layers, bs, dim)



class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, bidirectional=False):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size + bidirectional * hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size + bidirectional * hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        # self.W3 = nn.Linear(hidden_size, 1)

    def forward(self, query, values, mask):
        '''
        query: decoder's previous hidden state (bs, 1=n_layers, dim)
        values: encoder outputs (bs, seqlen, dim)
        mask: input mask (bs, seqlen)
        '''
        # Additive attention
        scores = self.V(torch.tanh(self.W1(query) + self.W2(values)))
        # pdb.set_trace()
        scores = scores.squeeze(2).unsqueeze(1) # bs, seqlen, 1 -> bs, 1 seqlen
        # pdb.set_trace()

        # Dot-Product Attention: score(s_t, h_i) = s_t^T h_i
        # Query [B, 1, D] * Values [B, D, M] -> Scores [B, 1, M]
        # scores = torch.bmm(query, values.permute(0,2,1))

        # Cosine Similarity: score(s_t, h_i) = cosine_similarity(s_t, h_i)
        # scores = F.cosine_similarity(query, values, dim=2).unsqueeze(1)

        # Mask out invalid positions.
        scores.data.masked_fill_(mask.unsqueeze(1) == 0, -float('inf'))
        # pdb.set_trace()

        # Attention weights
        alphas = F.softmax(scores, dim=-1)

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, values)

        # context shape: [B, 1, D], alphas shape: [B, 1, M]
        return context, alphas


class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_vocab_size, num_layers, dropout, bidirectional, bos_token_id):
        super(AttnDecoder, self).__init__()
        self.embedding = nn.Embedding(output_vocab_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size, bidirectional)
        self.gru = nn.GRU(
            hidden_size*2 + hidden_size*bidirectional, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout, 
            bidirectional=bidirectional,
        )
        self.bos_token_id = bos_token_id
        self.out = nn.Linear(hidden_size + hidden_size*bidirectional, output_vocab_size)
        self.bidirectional = bidirectional


    def forward(self, encoder_outputs, encoder_hidden, input_mask,
                target_tensor=None, length=10):
        # Teacher forcing if given a target_tensor, otherwise greedy.
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.tensor([[self.bos_token_id]]*batch_size).to(encoder_outputs).long()
        # decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(self.tgt_tokenizer.bos_token_id)
        decoder_hidden = encoder_hidden # nlayers, bs, dim. TODO: Consider bridge
        decoder_outputs = []

        # for i in range(max_len):
        if length is None:
            assert target_tensor is not None
            length = target_tensor.shape[1]
        for i in range(length):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs, input_mask)
            decoder_outputs.append(decoder_output)

            # if target_tensor is not None:
            if self.training and random.random() < self.teacher_forcing_p:
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                topv, topi = decoder_output.data.topk(1)
                decoder_input = topi.squeeze(-1)

        decoder_outputs = torch.cat(decoder_outputs, dim=1) # [B, Seq, OutVocab]
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden


    def forward_step(self, input, hidden, encoder_outputs, input_mask):
        '''
        input: decoder's input at current step (bs, 1, dim)
        hidden: decoder's previous hidden state (n_layers, bs, dim)
        encoder_outputs: (bs, seqlen, dim)
        input_mask: (bs, seqlen)
        '''
        # encoder_outputs: [B, Seq, D]
        if self.bidirectional:
            query = torch.cat([hidden[[-2], :, :], hidden[[-1], :, :]], dim=-1).permute(1, 0, 2) # [1, B, D*2] --> [B, 1, D*2]
        else:
            query = hidden[[-1], :, :].permute(1, 0, 2) # [1, B, D] --> [B, 1, D]
        # query = hidden.permute(1, 0, 2) # [1, B, D] --> [B, 1, D]
        context, attn_weights = self.attention(query, encoder_outputs, input_mask)
        embedded = self.embedding(input)
        attn = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(attn, hidden)
        output = self.out(output)
        # output: [B, 1, OutVocab]
        return output, hidden, attn_weights



class EncoderDecoder(pl.LightningModule):
    def __init__(self, hyperparams, hidden_size, num_encoder_layers: int, num_decoder_layers: int,
                 bidirectional, maxlen, dropout:float = 0.1, 
                 src_tokenizer_name=None, tgt_tokenizer_name=None, ):
        super(EncoderDecoder, self).__init__()
        if hasattr(hyperparams.tf, "code"):
            hyperparams.tf.message = init_tf_function(hyperparams.tf.code)['message']
            self.get_tf_p = init_tf_function(hyperparams.tf.code)['function']
        self.save_hyperparameters(hyperparams) if hyperparams is not None else None
        self.hyperparameters = hyperparams
        self.hidden_size = hidden_size
        # tokenizer
        self.src_vocab: Vocab = torch.load('data/vocab_english.pt', map_location='cpu')
        self.tgt_vocab: Vocab = torch.load('data/vocab_chinese.pt', map_location='cpu')
        self.src_tokenizer = Tokenizer(src_tokenizer_name, padding_side='right', maxlen=maxlen)
        self.tgt_tokenizer = Tokenizer(tgt_tokenizer_name, padding_side='right', maxlen=maxlen)
        self.encoder = EncoderRNN(
            input_vocab_size=self.src_tokenizer.vocab_size, 
            hidden_size=hidden_size, 
            num_layers=num_encoder_layers, 
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.decoder = AttnDecoder(
            hidden_size=hidden_size, 
            output_vocab_size = self.tgt_tokenizer.vocab_size,
            num_layers=num_decoder_layers, 
            dropout=dropout,
            bidirectional=bidirectional,
            bos_token_id=self.tgt_tokenizer.bos_token_id
            # teacher_forcing_p=teacher_forcing_p,
        )


        self.bidirectional = bidirectional
        self.maxlen = maxlen

        # criterion
        self.criterion = nn.NLLLoss(ignore_index=self.tgt_tokenizer.pad_token_id)


    def forward(self, inputs, input_mask, targets=None, length=None):
        #  bs, seqlen (long)

        encoder_outputs, encoder_hidden = self.encoder(inputs) # (bs, seqlen, dim), (n_layers, bs, dim)
        # pdb.set_trace()
        decoder_outputs, decoder_hidden = self.decoder(
            encoder_outputs, encoder_hidden, input_mask, targets, length=length)
        return decoder_outputs, decoder_hidden
    

    def create_mask(self, input_tensor, pad_token_id):
        # bs seqlen
        # if pad token, then 0
        return input_tensor != pad_token_id


    def shared_step(self, batch, batch_idx):
        import pdb
        self.decoder.teacher_forcing_p = self.get_tf_p(self.global_step)
        src, tgt = batch['src'], batch['trg'] # string
        self.batch_size = len(src)
        # bs, seqlen (long)
        src = self.src_tokenizer(src, padding=True, truncation=True, return_tensors="pt")
        tgt = self.tgt_tokenizer(tgt, padding=True, truncation=True, return_tensors="pt")
        input_tensor = src.to(self.device)
        target_tensor = tgt.to(self.device)
        input_mask = self.create_mask(input_tensor, pad_token_id=self.src_tokenizer.pad_token_id)

        return input_tensor, input_mask, target_tensor
    

    def training_step(self, batch, batch_idx):
        input_tensor, input_mask, target_tensor = self.shared_step(batch, batch_idx)
        decoder_outputs, decoder_hidden = self.forward(input_tensor, input_mask, target_tensor)

        loss = self.criterion(
        decoder_outputs.view(-1, decoder_outputs.size(-1)), # [B, Seq, OutVoc] -> [B*Seq, OutVoc]
        target_tensor.view(-1) # [B, Seq] -> [B*Seq]
        )

        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
        self.log('tf', self.decoder.teacher_forcing_p, logger=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
        lr = None
        if self.lr_schedulers():
            lr = self.lr_schedulers().get_last_lr()[0]
            # self.log('lr', lr, prog_bar=False, logger=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
        if self.global_step % self.trainer.print_every == 0 and self.global_rank == 0:
            if lr:
                logging.info(f'step: {self.global_step}, train_loss: {loss.item():.4f}, tf: {self.decoder.teacher_forcing_p:.2f}, lr: {lr:e} @ {datetime.now()}')
            else:
                logging.info(f'step: {self.global_step}, train_loss: {loss.item():.4f}, tf: {self.decoder.teacher_forcing_p:.2f} @ {datetime.now()}')

        return loss

    def validation_step(self, batch, batch_idx):
        input_tensor, input_mask, target_tensor = self.shared_step(batch, batch_idx)
        decoder_outputs, decoder_hidden = self.forward(input_tensor, input_mask, length=target_tensor.shape[1])
        loss = self.criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)), # [B, Seq, OutVoc] -> [B*Seq, OutVoc]
            target_tensor.view(-1) # [B, Seq] -> [B*Seq]
        )

        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
        if batch_idx % self.trainer.print_every == 0 and self.global_rank == 0:
            logging.info(f'\tval_loss: {loss.item():.4f} @ {datetime.now()}')
        return loss

    @torch.no_grad()
    def generate(self, src, do_sample=True, temperature=1.0, topk=None):
        
        src = self.src_tokenizer(src, padding=True, truncation=True, return_tensors="pt")
        src = src.to(self.device)
        input_mask = self.create_mask(src, pad_token_id=self.src_tokenizer.pad_token_id)

        decoder_outputs, decoder_hidden = self(src, input_mask, length=50)
        del decoder_hidden

        if do_sample :
            decoded_ids = []
            if topk:
                decoder_outputs, topi = decoder_outputs.topk(topk)
                del topi
                
            decoder_outputs /= temperature
            probs = torch.softmax(decoder_outputs, dim=-1) # bs, seqlen, dim
            for p in probs:
                d = torch.multinomial(p, 1) # (seqlen, 1)
                decoded_ids.append(d[: ,0])
            decoded_ids = torch.stack(decoded_ids) # bs, seqlen
            
        else:
            topv, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze(-1) # (bs, seqlen)

        batch_outputs = self.src_tokenizer.batch_decode(decoded_ids, self.tgt_vocab)
        return batch_outputs

    
    
    

    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(list(self.encoder.parameters())+list(self.decoder.parameters()), 
                                      **self.hyperparameters.optimizer.general)
                                    #   lr=self.learning_rate, weight_decay=self.weight_decay)
        # scheduler = MyScheduler(optimizer, dim_embed=self.hidden_size, 
        #                         warmup_steps=self.hyperparameters.scheduler.warmup_steps,
        #                         peak_lr=self.hyperparameters.scheduler.peak_lr,
        #                         verbose=self.hyperparameters.scheduler.verbose)
        # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=1)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optim_groups, )
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=1, threshold=0.5)
        returned = {
                "optimizer": optimizer,
                # "lr_scheduler": {
                #     "scheduler": scheduler,
                #     "monitor": self.hyperparameters.scheduler.monitor,
                #     "frequency": self.hyperparameters.scheduler.frequency,
                #     "interval": self.hyperparameters.scheduler.interval,
                # }, # dbg
            }

        
        return returned



# in case they will be used in the future, I keep them here. 
from torch.optim.lr_scheduler import _LRScheduler

class MyScheduler(_LRScheduler):
    # https://kikaben.com/transformers-training-details/
    def __init__(self, 
                 optimizer,
                 dim_embed: int,
                 warmup_steps: int,
                 peak_lr: float,
                 last_epoch: int=-1,
                 verbose: bool=False) -> None:

        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)
        self.peak_lr = peak_lr

        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self) -> float:
        lr = calc_lr(self._step_count, self.dim_embed, self.warmup_steps, self.peak_lr)
        return [lr] * self.num_param_groups

from math import log as ml
def calc_lr(step, dim_embed, warmup_steps, peak):
    # peak = 2.e-4
    const = dim_embed**(-0.5)
    p = ml(const / peak, warmup_steps)
    # return dim_embed**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))
    # return dim_embed**(-0.5) * min(step * warmup_steps**(-2.3), step**(-1.3)) # v4, 100 steps to 1.1e-4
    # return const * min(step * warmup_steps**(-p-1), step**(-p)) # v5, 500 steps to 1.1e-4
    return const * min(step * warmup_steps**(-p-1), step**(-p)) # v6, 500 steps to 2.e-4


# def get_tf_p(self):
#     # dbg
#     # return .5
#     # return max(1 - .1 * self.current_epoch, 0.8 - 0.02 * (self.current_epoch)-2, .0) # v2
#     # return {'message': "constant at 1", 'tf': 1.} # 
#     return {'message': "constant at 0.5", 'tf': .5} # 
#     # return {'message': "constant at 0.", 'tf': .0} # 
    
def init_tf_function(code):
    if code == 0:
        return {'function': lambda x: 1, 'message': "constant at 1"} 
    elif code == 1:
        return {'function': lambda x: .5, 'message': "constant at 0.5"} 
    elif code == 2:
        return {'function': lambda x: .0, 'message': "constant at 0"} 
    elif code == 3:
        return {'function': lambda x: max(1 - 0.5/1000 * x, .5), 'message': "start at 1, linearly decay to 0.5 in 1000 steps"} # 
    elif code == 4:
        return {'function': lambda x: max(.5 - 0.5/1000 * x, .0), 'message': "start at 0.5, linearly decay to 0 in 1000 steps"} # 
    elif code == 5:
        return {'function': lambda x: max(1 - 1/2000 * x, .0), 'message': "start at 1, linearly decay to 0 in 2000 steps"} # 

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
            return torch.tensor(encoded) # batched tokens
        
 
    def batch_decode(self, sequences, zh_vocab: Vocab):
        '''
        batch: bs, seqlen
        '''
        sequences = sequences.detach().cpu().numpy().tolist()
        decoded = []
        for seq in sequences:

            if self.eos_token_id in seq:
                eos_pos = seq.index(self.eos_token_id) # locate the first eos token
                seq = seq[1:eos_pos] # fetch all tokens bwtween bos and eos
            else:
                seq = seq[1:] # fetch all tokens bwtween bos and eos
            
            seq_d = zh_vocab.lookup_tokens(seq) # list[str]
            seq_d = ''.join(seq_d)
            decoded.append(seq_d)

        return decoded
    

###########################
def clamp_inf(t, clamp_max_only=True):
    '''
    Clamp the tensor values
    '''
    if torch.isinf(t).any():
        clamp_value = torch.finfo(t.dtype).max - 100
        if clamp_max_only:
            t = torch.clamp(t, max=clamp_value) # only clamp the positive infs
        else:
            t = torch.clamp(t, max=clamp_value, min=-clamp_value) # only clamp the positive infs
    return t
###########################