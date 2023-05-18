# Bahdanau-and-Transformer-on-NMT


## Steps to run:

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
