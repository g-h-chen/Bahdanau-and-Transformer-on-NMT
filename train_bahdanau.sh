


# teacher-forcing ratio
TOKENIZERS_PARALLELISM=true python train_bahdanau.py  --name improvement --tf_code 0 --version 0
TOKENIZERS_PARALLELISM=true python train_bahdanau.py  --name improvement --tf_code 1 --version 1
TOKENIZERS_PARALLELISM=true python train_bahdanau.py  --name improvement --tf_code 2 --version 2
TOKENIZERS_PARALLELISM=true python train_bahdanau.py  --name improvement --tf_code 3 --version 3
TOKENIZERS_PARALLELISM=true python train_bahdanau.py  --name improvement --tf_code 4 --version 4
TOKENIZERS_PARALLELISM=true python train_bahdanau.py  --name improvement --tf_code 5 --version 5

# hidden dimension
TOKENIZERS_PARALLELISM=true python train_bahdanau.py  --name dim  --version 0 --hidden_size 128
TOKENIZERS_PARALLELISM=true python train_bahdanau.py  --name dim  --version 1 --hidden_size 256
TOKENIZERS_PARALLELISM=true python train_bahdanau.py  --name dim  --version 2 --hidden_size 512
TOKENIZERS_PARALLELISM=true python train_bahdanau.py  --name dim  --version 3 --hidden_size 1024
TOKENIZERS_PARALLELISM=true python train_bahdanau.py  --name dim  --version 4 --hidden_size 2048