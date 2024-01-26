# NanoGPT-Loss-Stop-Analysis
GPT-2 is an OpenAI developped model using transformers aimed at NLP tasks. The repository will use the NanoGPT implementation, on the Tiny Stories dataset and analyze how a variety of different loss functions and early stopping techniques can improve the performance of the model.

## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3


## Prepare

The model uses Tiny Stories data set from huggingface,
[dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
to prepare the data into train.bin and val.bin for training:
```
python3 data/TinyStories/prepare.py
```

## Training

The model is using GPT-2 with OpenAI's weights, this can be seen in the model.py file. It is 124 Million parameters and will be trained on the Tiny Stories dataset.

Please run with at least one GPU, the approximate training time on one 2080ti is 4 hours:

Change the batch_size as necessary to avoid CUDA memory errors:

One GPU:
```
python3 train.python3 train.py --compile=False --batch_size=6 --eval_interval=50
```

2 GPUs, one node:
```
torchrun --standalone --nproc_per_node=2 train.py --compile=False --batch_size=5 --eval_interval=50
```

8 GPUs, multiple nodes (cluster environment):
```
Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

All Command Line arguments:
```
---compile=False
--batch_size=6
--eval_interval=50 # Number of steps before evaluating the model's performance on validation data
--init_from='resume' # whether or not to "resume" training from previous checkpoint of model
```

To train with MSE loss function, use --loss_func="mse" command above.

## Evaluation

The output of training the model will show the train and val loss, for each iteration.

To get the perplexity of the model trained run:
```
python3 test_perplexity.py --loss_func="mse"
```

## Sampling Text

To sample text from the model after training it:
```
python3 sample.py --start="Once upon a time," --num_samples=5 --max_new_tokens=100
```

To sample from a model that was trained using mse loss, use --loss_func="mse" on the command above


