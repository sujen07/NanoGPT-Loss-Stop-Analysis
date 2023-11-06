# NanoGPT-Loss-Stop-Analysis
using various loss functions and early stopping techniques to improve nanoGPT performance. Uses GPT2

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

The model uses openwebtext-10k data set from huggingface,
[dataset](https://huggingface.co/datasets/stas/openwebtext-10k)
to prepare the data into train.bin and val.bin for training:
```
python3 data/openwebtext/prepare.py
```

## Training

The model is using GPT-2 with OpenAI's weights, this can be seen in the model.py file. It is 124 Million parameters and will be trained on the openweb-10k dataset.

Please run with at least one GPU, the approximate training time on one 2080ti is 4 hours:

Change the batch_size as necessary to avoid CUDA memory errors:
```
python3 train.python3 train.py --compile=False --batch_size=6
```

