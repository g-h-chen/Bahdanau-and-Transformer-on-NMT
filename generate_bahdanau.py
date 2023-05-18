import torch
from models.rnn_sample import EncoderDecoder
from omegaconf import OmegaConf
import logging
import os
import json
from tqdm import trange
from models.bahdanau import EncoderDecoder
import pdb


def inputs_reader():
    input_pth = 'data/val.jsonl'
    texts = []
    chinese = []
    with open(input_pth, 'r+', encoding='utf-8') as f:
        for item in f:
            item = json.loads(item)
            texts.append(item['english'])
            chinese.append(item['chinese'])
    
    return texts, chinese

def outputs_writer(info_dict: dict, fp: str):
    # lens = 
    keys = list(info_dict.keys())
    length = min(map(len, [info_dict[k] for k in keys]))

    with open(fp, 'w', encoding='utf-8') as f:
        for i in range(length):
            line = {k: info_dict[k][i] for k in keys}
            jout = json.dumps(line, ensure_ascii=False) + '\n'
            f.write(jout)
    print(f'output to {fp}')



def load_config(config_pth):
    # config_pth = 'utils/transformer_config.yaml'
    config = OmegaConf.load(config_pth)
    return config

def load_model(config, ckpt_pth):    
    logging.info(f'loading model from {ckpt_pth}')
    model = EncoderDecoder(config, **config.EncoderDecoder,)
    if ckpt_pth is not None:
        sd = torch.load(ckpt_pth, map_location='cpu')['state_dict']
        model.load_state_dict(sd)
    return model.to(device)

def generate_files(name, version, best_or_last='best', bs=0, do_sample=False):
    # ckpt_pth = ckpt_pth if ckpt_pth else None
    bs = int(bs)
    ckpt_dir = f'output/{name}/version_{version}'
    if best_or_last == 'best':
        for f in os.listdir(ckpt_dir):
            if f.startswith('epoch'):
                ckpt_pth = os.path.join(ckpt_dir, f)
                break
    elif best_or_last == 'last':
        ckpt_pth = os.path.join(ckpt_dir, 'last.ckpt')
    config_pth = f'logs/{name}/version_{version}/hparams.yaml'

    config = load_config(config_pth)
    model = load_model(config, ckpt_pth)
    model.eval()

    inputs, chinese = inputs_reader()
    outputs = []
    # bs = 160
    # bs = 30
    print(f'bs={bs}')
    for i in trange(len(inputs)//bs+1):
        start, end = i*bs, (i+1) * bs
        batch_inputs = inputs[start:end]
        batch_outputs = model.generate(batch_inputs, do_sample=do_sample, temperature=.01, topk=20)

        # batch_inputs = model.src_tokenizer(batch_inputs, padding=True, truncation=True, return_tensors="pt")
        # batch_inputs = batch_inputs.to(device)
        # input_mask = create_mask(batch_inputs, pad_token_id=model.src_tokenizer.pad_token_id)

        # decoder_outputs, decoder_hidden = model(batch_inputs, input_mask, length=50)
        # pdb.set_trace()
        # if do_sample :
        #     temperature = 3
        #     decoder_outputs /= temperature
        #     prob = torch.softmax(decoder_outputs, dim=-1)
        #     decoded_ids = torch.multinomial(prob, 1)[:, :, 0]
        # else:
        #     topv, topi = decoder_outputs.topk(1)
        #     decoded_ids = topi.squeeze(-1) # (bs, seqlen)

        # batch_outputs = model.src_tokenizer.batch_decode(decoded_ids, model.tgt_vocab)


        outputs.extend(batch_outputs)

    
    info = {'english': inputs, 'chinese': chinese, 'prediction': outputs}
    # info = {'prediction': outputs}
    outputs_writer(info, os.path.join(ckpt_dir, f'{name}_v{version}_prediction.jsonl'))



def interact(name=None, version=None, best_or_last='best', do_sample=False):
    # ckpt_pth = ckpt_pth if ckpt_pth else None
    ckpt_dir = f'output/{name}/version_{version}'
    if best_or_last == 'best':
        for f in os.listdir(ckpt_dir):
            if f.startswith('epoch'):
                ckpt_pth = os.path.join(ckpt_dir, f)
                break
    elif best_or_last == 'last':
        ckpt_pth = os.path.join(ckpt_dir, 'last.ckpt')
    config_pth = f'logs/{name}/version_{version}/hparams.yaml'


    config = load_config(config_pth)
    model = load_model(config, ckpt_pth)


    model.eval()
    s, t = False, 1.
    while True:
        inputs = input('input english: ')
        if inputs in ['q', 'quit', 'exit']:
            break
        print(f'do_sample={s}, temp={t}')

        outputs = model.generate([inputs], do_sample=do_sample, temperature=.01, topk=20)
        print(outputs)



    




if __name__ == '__main__':
    import sys
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    interact(name=sys.argv[1], version=sys.argv[2], best_or_last=sys.argv[3])
    # generate_files(name=sys.argv[1], version=sys.argv[2], best_or_last=sys.argv[3], bs=sys.argv[4])