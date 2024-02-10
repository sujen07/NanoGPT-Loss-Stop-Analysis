import os
from itertools import product

learning_rates = [0.000006,0.0006,0.06]
dropouts = [0.0, 0.1, 0.2]
n_layers = [8, 12, 16]


params = list(product(learning_rates, dropouts, n_layers))

counter = 0
for lr, dropout, n_layer in params:
    command = f'python3 train.py --compile=False --batch_size=4 --eval_interval=50 --learning_rate={lr:.9f} --min_lr={lr/10:.9f} --dropout={dropout} --n_layer={n_layer} --ckpt_name=ckpt{counter}.pt'
    #print(command)
    os.system(command)
    counter += 1