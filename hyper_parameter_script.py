import os
from itertools import product

learning_rates = [0.000006,0.0006,0.06]
dropouts = [0.0, 0.1, 0.2]
n_layers = [8, 12, 16]


params = list(product(learning_rates, dropouts, n_layers))

counter = 0
for lr, dropout, n_layer in params:
    command = f'python3 train.py --compile=False --wandb_log=True --batch_size=4 --max_iters=50 --eval_interval=50 --loss_func="squentropy" --learning_rate={lr:.9f} --min_lr={lr/10:.9f} --dropout={dropout} --n_layer={n_layer} --ckpt_name=ckpt{counter}.pt'
    #print(command)
    os.system(command)
    
    move_command = f'scp out/ckpt{counter}.pt skancherla@dsmlp-login.ucsd.edu:/dsmlp/workspaces-fs04/DSC180A_FA23_A00/teams/b13/group1/ckpt{counter}.pt'
    os.system(move_command)
    counter += 1
    break