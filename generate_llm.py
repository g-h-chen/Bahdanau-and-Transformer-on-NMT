import sys
sys.path.insert(0, '.')
import os
import shortuuid
import torch
from tqdm import tqdm, trange
from utils import __ALL_MODELS__
from utils.llm_utils import (
    inputs_reader, 
    get_output_path,
    results_writer, 
    load_model_and_tokenizer,
    get_model_outputs,
    logging,
)

from argparse import ArgumentParser
from omegaconf import OmegaConf
import logging 



@torch.no_grad()
def generate(
    config_dir = 'models/chinese-Alpaca-7B',
    model_id = 'chinese-alpaca-7b',

    prompt = '',
    generation_config={}, # for huggingface generation_config

    batch_size = 10,

    device = 'cuda',
    n_gpus = 2,
    precision = 'fp16',

    answer_id = 'auto',
    input_pth = 'data/question_zh.jsonl',
    output_dir_or_file = 'output',
    private = True,
    *args,
    **kwargs,
):
    model_config = {k: v for k, v in locals().items()} # for meta_data
    model_config.pop('generation_config')
    model_config.update(**generation_config)

    # avoid leaking private path
    if private:
        model_config['config_dir'] = ''

    logging.info(f'using dataset: {input_pth}')
    # output_path = get_output_path(output_dir_or_file, model_id)
    output_path = f'output/llm/version_{model_id}/llm_v{model_id}_prediction.jsonl'
    os.makedirs(f'output/llm/version_{model_id}', exist_ok=True)
    logging.info('output to: {}'.format(output_path))

    # load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_id, device, n_gpus, precision, config_dir)
    generation_config['pad_token_id'] = tokenizer.pad_token_id
    
    # read inputs
    english, chinese, _, _, _ = inputs_reader(input_pth, model_config, prompt)  
    # length = 8000
    # english = english[:length]  
    # chinese = chinese[:length]  
    
    predictions = [] # extended as iteration goes. Extension happens in utils/get_model_outputs().

    n_iters = len(english)//batch_size + 1
    for idx in trange(n_iters):
        start, end = idx * batch_size, (idx+1) * batch_size
        if start >= len(english):
            break

        # extend texts
        predictions = get_model_outputs(
            model_id, 
            model, 
            tokenizer, 
            english[start:end], 
            generation_config, 
            predictions, 
            device
        )
        
    del model; torch.cuda.empty_cache()

    results = []
    for i, prediction in enumerate(predictions):
        results.append({
            'english': english[i],
            'chinese': chinese[i],
            'prediction': prediction,
            # 'answer_id': shortuuid.uuid() if answer_id=='auto' else qids[i],
            # 'model_id': model_id,
            # 'category': categories[i],
            # 'lang': langs[i],
            # 'metadata': metas[i]
        })

    results_writer(results, model_id, output_path)
    return results


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config_pth', default=None)
    parser.add_argument('--eval_models', nargs="+", type=str, choices=__ALL_MODELS__)

    parser.parse_args()

    arg = parser.parse_args()
    return arg


def main(arg):
    config = OmegaConf.load(arg.config_pth)
    eval_models = arg._get_kwargs()[1][1] # TODO: make it nicer
    print(f'generating outputs from\n{eval_models}')

    # call generate() function for each model
    for model_id in __ALL_MODELS__:
        if model_id in eval_models:
            print('-'*20+f'generating {model_id}'+'-'*20)
            results = generate(**config[model_id])



if __name__ == '__main__':
    arg = get_args()
    main(arg)
