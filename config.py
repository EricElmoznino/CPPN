import torch.nn as nn

resolution = [1000, 2000]
rgb = True
z_dim = 1
z_scale = 10.0
n_layers = 9
layer_sizes = 17
activations = nn.Tanh()
initialization = {'type': 'normal', 'stddev': 1}
