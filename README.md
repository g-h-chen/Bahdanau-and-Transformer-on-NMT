# Bahdanau-and-Transformer-on-NMT

## Introdcution
This repo is for a project of DDA4220 taught by <a href="http://www.zhangruimao.site/" target="_blank">Prof. Zhang Ruimao</a> during AY22-23 Spring @CUHKSZ.

<!-- [Prof. Zhang Ruimao](http://www.zhangruimao.site/) during AY22-23 Spring @CUHKSZ.  -->

## Contribution
### Code
We implemented Bahdanau attention machanism from scratch and a vanilla version of Transformer using `nn.Transformer`. Both models are implemented in PyTorch.

For Bahdanau attention, 
* mainly referring to [https://github.com/mhauskn/pytorch_attention/tree/main](https://github.com/mhauskn/pytorch_attention/tree/main)
* supports bidirection, multiple layers of encoders and decoders

For vanilla Transformer,
* uses `nn.Transformer` as the code base of our model
* supports fast inference by overriding `forward` methods of decoder layers. Although `model.generate(..., use_cache=True)` in HuggingFace's inference pipeline does the same thing, we found that they have not implemented the vanilla Transformer (Please raise an issue if I am wrong!). Most importantly, we implemented it for the purpose of practice.

For both models,
* support training on multiple GPUs via `pytorch_lightning`
* support fp16 training via `pytorch_lightning`
 
All codes are integrated/written by the owner of this repo. Reach me via [my email](mailto:guimingchen@link.cuhk.edu.cn) should you have any suggestion or questions.

Please also check out [LLMZoo](https://github.com/FreedomIntelligence/LLMZoo/tree/main) and play around with Phoenix and Chimera!

Codes for evaluation are taken from [here](https://github.com/mjpost/sacrebleu).

### Experiments
We completed a report aiming to compare the ability of three (types of) attention-based models:
* Bahdanau RNN
* vanilla Transformer
* LLMs (Phoenix and Chimera)

Here are our major findings:
- Badanau RNN
	- Bahdanau RNN tends to be **lazy** during training (sec 5.2)
	- Under our settings, the best teacher forcing ratio is found to be 0.5 during training (sec 5.3)
	- A network being wide yet too shallow results in a slight decrease in performance (sec 5.4)

- Vanilla Transformer (sec 6.3)
	- Deeper ≠ better (Table 3)
	- The model benefits from a slightly unbalanced structure. Given a total number of layers $N = E + D = $(\#encoder layers) + (\#decoder layers), $D>E$ often results in a better performance, indicating that **decoder** is a more powerful component in the model. (Table 4)

- LLMs (Phoenix and Chimera) (sec 7.2)
	- Phoenix outperforms Chimera. No surprise since the target language is Chinese, which is a non-Latin language, but Chimera is trained solely on Latin language while Phoenix is trained on both Latin and non-Latin language. 
	- Instruction tuning is critical. All instrution-tuned models outperform their counterparts.



## Steps to run

1. Install packages
    ```bash
    pip install -r requirements.txt
    ```

2. Prepare training and validation data
    ```bash
    python utils/combine_parts.py   # combine training file parts
    python utils/process.py		    # does nothing but rename the keys in the json files
    python utils/build_vocab.py		# builds vocab files that can be reused
    python utils/make_ans_txt.py	# extracts the target language of validation dataset and stores it in the format that is required by the evaluation part.
    ```
    For Chinese (target), we split each character as a token. For English (source), we use `bert-base-uncased` tokenizer, but only words that appear at least twice will be added to the vocab file. 


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
    bash evaluate.sh
    ```
