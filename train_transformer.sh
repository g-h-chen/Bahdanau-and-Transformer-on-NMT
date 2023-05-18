

# |delta| = 0
TOKENIZERS_PARALLELISM=true python train_transformer.py --gpus 1 --name depth   --n_enc 5 --n_dec 5     --version 5
TOKENIZERS_PARALLELISM=true python train_transformer.py --gpus 1 --name depth   --n_enc 10 --n_dec 10   --version 2
TOKENIZERS_PARALLELISM=true python train_transformer.py --gpus 1 --name depth   --n_enc 15 --n_dec 15   --version 4
TOKENIZERS_PARALLELISM=true python train_transformer.py --gpus 1 --name depth   --n_enc 20 --n_dec 20   --version 3

# |delta| = 5
TOKENIZERS_PARALLELISM=true python train_transformer.py --gpus 1 --name depth   --n_enc 10 --n_dec 5    --version 11
TOKENIZERS_PARALLELISM=true python train_transformer.py --gpus 1 --name depth   --n_enc 15 --n_dec 10   --version 15
TOKENIZERS_PARALLELISM=true python train_transformer.py --gpus 1 --name depth   --n_enc 20 --n_dec 15   --version 9

TOKENIZERS_PARALLELISM=true python train_transformer.py --gpus 1 --name depth   --n_dec 10 --n_enc 5    --version 6
TOKENIZERS_PARALLELISM=true python train_transformer.py --gpus 1 --name depth   --n_dec 15 --n_enc 10   --version 14
TOKENIZERS_PARALLELISM=true python train_transformer.py --gpus 1 --name depth   --n_dec 20 --n_enc 15   --version 8

# |delta| = 10
TOKENIZERS_PARALLELISM=true python train_transformer.py --gpus 1 --name depth   --n_enc 15 --n_dec 5    --version  10
TOKENIZERS_PARALLELISM=true python train_transformer.py --gpus 1 --name depth   --n_enc 20 --n_dec 10   --version  17
TOKENIZERS_PARALLELISM=true python train_transformer.py --gpus 1 --name depth   --n_dec 15 --n_enc 5    --version  7
TOKENIZERS_PARALLELISM=true python train_transformer.py --gpus 1 --name depth   --n_dec 20 --n_enc 10   --version  16

# |delta| = 15
TOKENIZERS_PARALLELISM=true python train_transformer.py --gpus 1 --name depth   --n_enc 20 --n_dec 5   --version  13
TOKENIZERS_PARALLELISM=true python train_transformer.py --gpus 1 --name depth   --n_dec 20 --n_enc 5   --version  12