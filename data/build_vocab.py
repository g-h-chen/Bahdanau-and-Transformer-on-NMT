# https://blog.csdn.net/zhaohongfei_358/article/details/126175328?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168285041616800213026289%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168285041616800213026289&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-1-126175328-null-null.142^v86^control_2,239^v2^insert_chatgpt&utm_term=Pytorch%E4%B8%AD%20nn.Transformer&spm=1018.2226.3001.4187

import json
import torch

from tqdm import tqdm

from torchtext.vocab import build_vocab_from_iterator
from tokenizers import Tokenizer


# def en_tokenizer(line):
#     """
#     定义英文分词器，后续也要使用
#     :param line: 一句英文句子，例如"I'm learning Deep learning."
#     :return: subword分词后的记过，例如：['i', "'", 'm', 'learning', 'deep', 'learning', '.']
#     """
#     # 使用bert进行分词，并获取tokens。add_special_tokens是指不要在结果中增加‘<bos>’和`<eos>`等特殊字符
#     return tokenizer.encode(line, add_special_tokens=False).tokens

def yield_en_tokens(train_data_pth):
    """
    每次yield一个分词后的英文句子，之所以yield方式是为了节省内存。
    如果先分好词再构造词典，那么将会有大量文本驻留内存，造成内存溢出。
    """
    # file = open(en_filepath, encoding='utf-8')
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

    with open(train_data_pth, 'r') as f:
        for line in tqdm(f, desc="english vocab"):
            line = json.loads(line)['english']
            # "I'm learning Deep learning." -> ['i', "'", 'm', 'learning', 'deep', 'learning', '.']
            yield tokenizer.encode(line, add_special_tokens=False).tokens 
            # yield en_tokenizer(line)


def zh_tokenizer(line: str):
    """
    定义中文分词器
    :param line: str，例如：机器学习
    :return: list，例如['机','器','学','习']
    """
    return list(line.strip().replace(" ", ""))


def yield_zh_tokens(train_data_pth):
    with open(train_data_pth, 'r') as f:
        for line in tqdm(f, desc="chinese vocab"):
            line = json.loads(line)['chinese']
            yield list(line.strip().replace(" ", "")) # "机器学习" ->  ['机','器','学','习']


def build_zh_vocab(zh_vocab_file='data/vocab_chinese.pt', train_data_pth='data/train.jsonl'):
    zh_vocab = build_vocab_from_iterator(
        yield_zh_tokens(train_data_pth),
        min_freq=1,
        specials=["<s>", "</s>", "<pad>", "<unk>"],
    )
    zh_vocab.set_default_index(zh_vocab["<unk>"])
    torch.save(zh_vocab, zh_vocab_file)



def build_en_vocab(en_vocab_file='data/vocab_english.pt', train_data_pth='data/train.jsonl'):
    en_vocab = build_vocab_from_iterator(
        # 传入一个可迭代的token列表。例如[['i', 'am', ...], ['machine', 'learning', ...], ...]
        yield_en_tokens(train_data_pth),
        # 最小频率为2，即一个单词最少出现两次才会被收录到词典
        min_freq=2,
        # 在词典的最开始加上这些特殊token
        specials=["<s>", "</s>", "<pad>", "<unk>"],
    )
    # 设置词典的默认index，后面文本转index时，如果找不到，就会用该index填充
    en_vocab.set_default_index(en_vocab["<unk>"])
    # 保存缓存文件

    torch.save(en_vocab, en_vocab_file)

if __name__ == '__main__':
    print('It may take a while to build vocab files from training data. Please wait......')
    build_en_vocab(train_data_pth='data/train.jsonl')
    build_en_vocab(train_data_pth='data/train.jsonl')