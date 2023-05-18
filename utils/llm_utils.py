import json
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

import string


import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def timeit(func):
    from functools import wraps
    import time
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # print(f'/Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        print(f'Loaded in {total_time:.4f} seconds')
        return result
    return timeit_wrapper


@timeit
def load_model_and_tokenizer(model_id, device, n_gpus, precision, config_dir):
    print(f'loading {model_id}, this might take several minutes...')
    hf_model_config = {"pretrained_model_name_or_path": config_dir, 'trust_remote_code': True, 'low_cpu_mem_usage': True}
    hf_tokenizer_config = {"pretrained_model_name_or_path": config_dir, 'padding_side': 'left', 'trust_remote_code': True}

    if device == "cpu":
        pass

    elif device == "cuda":
        assert precision == 'fp16', 'Only supports fp16 for now'

        if precision == 'fp16':
            hf_model_config.update({"torch_dtype": torch.float16})

        if precision == 'int8':
            if n_gpus != "auto" and int(n_gpus) != 1:
                print("8-bit weights are not supported on multiple GPUs. Revert to use one GPU.")
            hf_model_config.update({"torch_dtype": torch.float16, "load_in_8bit": True, "device_map": "auto"})
        else:
            if n_gpus == "auto":
                hf_model_config["device_map"] = "auto"
            else:
                n_gpus = int(n_gpus)
                if n_gpus != 1:
                    hf_model_config.update({
                        "device_map": "auto",
                        "max_memory": {i: "13GiB" for i in range(n_gpus)},
                    })
    elif device == "mps":
        # TODO:  may be bugs here
        hf_model_config.update({"torch_dtype": torch.float16})
        # Avoid bugs in mps backend by not using in-place operations.
        # replace_llama_attn_with_non_inplace_operations()
    else:
        raise ValueError(f"Invalid device: {device}")

    # load model and tokenizer
    if model_id in ['vicuna-7b-v1.1', 'vicuna-13b-v1.1', 'chinese-vicuna-13b']:
        hf_tokenizer_config['use_fast'] = False
    else:
        hf_tokenizer_config['use_fast'] = True # hf default
    tokenizer = AutoTokenizer.from_pretrained(**hf_tokenizer_config)
    
    # hf_model_config['low_cpu_mem_usage'] = True
    if model_id in ['chatglm-6b']:
        model = AutoModel.from_pretrained(**hf_model_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(**hf_model_config)
            

    # calling model.cuda() mess up weights if loading 8-bit weights
    if device == "cuda" and n_gpus == 1 and not precision=='int8':
        model.to("cuda")

    elif device == "mps":
        model.to("mps")

    # if (device == "mps" or device == "cpu") and precision=='int8':
    #     compress_module(model)


    # set pad_token_id to eos_token_id if there is no pad_token_id
    if not tokenizer.pad_token_id and tokenizer.eos_token_id is not None:
        print('warning: No pad_token in the config file. Setting pad_token_id to eos_token_id')
        tokenizer.pad_token_id = tokenizer.eos_token_id
        assert tokenizer.pad_token_id == tokenizer.eos_token_id

    model.eval()
    return model, tokenizer



def inputs_reader(input_pth, config_dict, prompt=''):
    english = []
    chinese = []
    qids = []
    categories = []
    langs = []
    metas = []

    with open(input_pth, 'r+', encoding='utf-8') as f:
        for item in f:
            item = json.loads(item)
            if prompt != '':
                english.append(add_prompt(item['english'], prompt))
                chinese.append(item['chinese'])
            else:
                raise NotImplementedError
            # qids.append(item['question_id'])
            # categories.append(item.get('category', ''))
            # langs.append(item.get('lang', ''))

            # meta_data = item.get('meta_data', {})
            # meta_data.update(config_dict)
            # metas.append(meta_data)
    
    return english[:], chinese[:], categories, langs, metas

def get_output_path(output, model_id):
    if '.jsonl' in output.split('/')[-1]: # already a file path
        pth = output
    else: # a directory
        os.makedirs(output, exist_ok=True)
        pth = os.path.join(output, f'answer_{model_id}.jsonl')
    return pth

def results_writer(results: list, model_name: str, pth):
    
    with open(pth, 'w', encoding='utf-8') as f:
        for r in results:
            jout = json.dumps(r, ensure_ascii=False) + '\n'
            f.write(jout)
    print(f'results for {model_name} written to {pth}')


def add_prompt(s, prompt=''):
    r'''
    Note that if a prompt is given, there must be one and only one named index "{question}" in the string.
    Otherwise an assertion error will be thrown.

    Here is an example of a prompt:

        Please answer the following code in Python. No explanation is needed:\n\n{question}

    Prompts can be configured in the "prompt" entry of each model in utils/config.yaml.
    '''
    # Chimera, Phoenix
    # "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\nHuman: <s>{question}</s>Assistant: <s>"

    # Chinese Alpaca  
    # "Below is an instruction that describes a certain task. Write a response that appropriately completes the request.\n{question}"
    
    checkPlaceHolder(prompt)
    return prompt.format(question=s)
    

def checkPlaceHolder(s):
    parsed = list(string.Formatter().parse(s))
    assert parsed[0][1] == 'question', "the placeholder should be named {{question}}, found {}".format(parsed[1])
    return

def get_model_outputs(model_id, model, tokenizer, prompts, generation_config, texts, device):
    # inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
    # outputs = model.generate(inputs, **generation_config)
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, **generation_config)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    for i, (d, p) in enumerate(zip(decoded, prompts)):
        d = post_process(model_id, d, p, i)
        texts.append(d)
        # print(d)
        # print('-----------------')
    return texts

def post_process(model, r: str, p: str = '', i=0):
    '''
    Arguments:
        `r` (str): response from the model.
        `p` (str): prompt defined in utils/config.yaml and added to the model input. Default ''.

    Returns:
        `r` (str): processed response from the model.
    '''
    
    if model in ['chinese-alpaca-7b', 'chinese-alpaca-13b', 'belle-7b-2m', 'vicuna-7b-v1.1', 'vicuna-13b-v1.1']:
        r = r.strip().replace(p, '', 1).strip() 
        
    elif model in ['chimera-chat-7b', 'chimera-inst-chat-7b', 'chimera-chat-13b', 'chimera-inst-chat-13b']:
        p = p.replace('<s>', '').replace('</s>', '') # get the prompts without sos and eos
        r = r.strip().replace(p, '', 1).strip()
        try:
            r = re.search(r'Assistant: (.+)$', r, flags=re.S).group(1).strip() # filter for chat dataset
        except:
            pass
    
    elif model in ['phoenix-chat-7b', 'phoenix-inst-chat-7b']:
        p = p.replace('<s>', '').replace('</s>', '') # get the prompts without sos and eos
        r = r.strip().replace(p, '', 1).strip()

    elif model in ['chatglm-6b']:
        r = r.strip()
        r = r.replace("[[训练时间]]", "2023年")
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            ["\?", "？"],
        ]
        for item in punkts:
            r = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], r)
            r = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], r)
        r = r.strip().replace(p, '', 1).strip()

    elif model in ['chinese-vicuna-13b']:
        r = r.split("### Response:")[1].strip().replace('�','')

    elif model in ['vicuna-7b-v0', 'vicuna-13b-v0',]:
        r = r.split("###")[4].replace('Assistant:', '', 1).strip()
    
    elif model in ['moss-moon-003-sft', 'moss-moon-003-sft-plugin']:
        p = p.replace('<eoh>', '') # get the prompts without sos and eos
        r = r.strip().replace(p, '', 1).strip()

    # elif model in ['your model_id1', 'your_model_id_2']:
    #     # processing steps
    #     r = r.strip().replace(p, '', 1).strip() 

    else:
        if i==0:
            logging.warning('no post processing step is used. Please check post_process() in utils/utils.py ')
        

    return r