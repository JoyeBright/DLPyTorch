import torch
import random
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Controlling sources of randomness
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

config_WOAttention = {

"max_length_data":15, #Selecting only sentences that their length is less than 15
"teacher_forcing_ratio": 0.5,
"max_length": 80, #of model

"SOS_token": 0,
"EOS_token": 1,
"learning_rate": 0.01,
"n_iters":3000,

"hidden_size":256,

"device": device,

"print_every":500,
"plot_every": 500,

"training_loss": "output/training_loss",


}
