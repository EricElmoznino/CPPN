import torch.nn as nn

resolution = [2880, 5120]
rgb = True
z_dim = 2
z_scale = 1.0
n_layers = 9
layer_sizes = 17
activations = nn.Tanh()
initialization = {'type': 'normal', 'stddev': 1}

video_length = 60
video_z_cycles = [2, 3]
