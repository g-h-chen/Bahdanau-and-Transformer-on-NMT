# Bahdanau-and-Transformer-on-NMT

## Introdcution
This repo is for a project of DDA4220 taught by <a href="http://www.zhangruimao.site/" target="_blank">Prof. Zhang Ruimao</a> during AY22-23 Spring @CUHKSZ.

<!-- [Prof. Zhang Ruimao](http://www.zhangruimao.site/) during AY22-23 Spring @CUHKSZ.  -->

## Contribution
We implemented Bahdanau attention machanism from scratch and a vanilla version of Transformer using `nn.Transformer`. Both models are implemented in PyTorch.

For Bahdanau attention, 
* mainly referring to [https://github.com/mhauskn/pytorch_attention/tree/main](https://github.com/mhauskn/pytorch_attention/tree/main)
* supports bidirection, multiple layers of encoders and decoders

For vanilla Transformer,
* uses `nn.Transformer` as the code base of our model.
* supports fast inference by overriding `forward` methods of decoder layers. Although `model.generate(..., use_cache=True)` in HuggingFace's inference pipeline does the same thing, we found that they have not implemented the vanilla Transformer (Please raise an issue if I am wrong!). Most importantly, we implemented it for the purpose of practice.

For both models,
* support training on multiple GPUs via `pytorch_lightning`
* support fp16 training via `pytorch_lightning`
 
All codes are written by the owner of this repo. Reach me via [my email](mailto:guimingchen@link.cuhk.edu.cn) should you have any suggestion or questions.


## Steps to run

1. Install packages:
    ```bash
    pip install -r requirements.txt
    ```


2. Modify the keys in the original .json file and build vocab file from training data.
    ```bash
    python data/process.py
    python data/build_vocab.py
    ```
    For Chinese, we split each character as a token. For English, we use `bert-base-uncased` tokenizer, but only words that appear at least twice will be added to the vocab file. 


3. Experiment: Bahdanau RNN
    ```bash
    bash train_bahdanau.sh      # train
    bash generate_bahdanau.sh   # inference  
    ```


4. Experiment: vanilla Transformer
    ```bash
    bash train_transformer.sh       # train
    bash generate_transformer.sh    # inference
    ```

5. Experiment: Chimera and Phoenix
    ```bash
    python generate_llm.py --config_pth utils/llm_config.yaml --eval_models chimera-chat-7b chimera-inst-chat-7b chimera-chat-13b phoenix-chat-7b phoenix-inst-chat-7b chimera-inst-chat-13b
    ```

5. Evaluation
    ```bash
    python data/make_ans_txt.py
    bash evaluate.sh
    ```
