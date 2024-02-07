#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from contextlib import nullcontext
import tiktoken
from model import GPTConfig, GPT

init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
dataset='TinyStories'
loss_func = 'cross_entropy'
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------
ckpt_name = 'ckpt.pt'
if loss_func == 'mse':
    ckpt_name = 'ckpt_mse_loss.pt'


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load the validation data
data_dir = os.path.join('data', dataset)
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
print(data_dir)

# Batch generation function for the validation set
def get_val_batch():
    ix = torch.randint(len(val_data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((val_data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((val_data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Perplexity calculation function
def calculate_perplexity(model, device, num_batches):
    model.eval()
    total_loss = 0
    total_count = 0

    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets = get_val_batch()
            with ctx:
                logits, loss = model(inputs, targets)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Reshape log_probs and targets for calculating loss
            # Flatten targets to [batch_size * sequence_length]
            targets = targets.view(-1)
            # Flatten log_probs to [batch_size * sequence_length, vocab_size]
            log_probs = log_probs.view(-1, log_probs.size(-1))

            # Calculate the loss
            loss = F.nll_loss(log_probs, targets, reduction='sum')
            
            total_loss += loss.item()
            total_count += targets.numel()  # Count total number of target tokens

    average_loss = total_loss / total_count
    perplexity = np.exp(average_loss)
    return perplexity
# Configuration
block_size = 256  # adjust to match model's expected input size
batch_size = 4    # adjust based on your GPU's memory

# Load model from checkpoint
# ... (use your existing model loading code here) ...
ckpt_path = os.path.join(out_dir, ckpt_name)
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf, loss_func)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

# Calculate Perplexity
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
num_batches = 10  # You can adjust this to the number of batches you want to use for calculation
perplexity = calculate_perplexity(model, device, num_batches)
print(f"Validation Perplexity: {perplexity}")

