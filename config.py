import torch.nn as nn

resolution = [1334, 750]
rgb = True
z_dim = 2
z_scale = 1
n_layers = 10
layer_sizes = 10
activations = nn.Tanh()
initialization = {'type': 'normal', 'stddev': 1}

video_length = 60
video_z_cycles = [2, 3]
