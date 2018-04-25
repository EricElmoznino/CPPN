import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import functional as tr
import numpy as np


class CPPN(nn.Module):

    def __init__(self, resolution, rgb, z_dim, z_scale, n_layers, layer_sizes, activations, initialization):
        super().__init__()
        self.eval()

        if type(layer_sizes) == list:
            assert  n_layers == len(layer_sizes)
        else:
            layer_sizes = [layer_sizes for _ in range(n_layers)]
        if type(activations) == list:
            assert n_layers == len(activations)
        else:
            activations = [activations for _ in range(n_layers)]

        self.resolution = resolution
        self.rgb = rgb
        self.z_dim = z_dim
        self.z_scale = z_scale
        self.coordinates = self.generate_coordinates()

        layers = [nn.Linear(3 + z_dim, layer_sizes[0], bias=False), activations[0]]
        for i in range(1, n_layers):
            layers += [nn.Linear(layer_sizes[i - 1], layer_sizes[i]), activations[i]]
        layers += [nn.Linear(layer_sizes[-1], 3 if rgb else 1), nn.Sigmoid()]
        self.layers = nn.Sequential(*layers)

        self.initialize_params(initialization)

    def forward(self, z):
        inputs = np.hstack([self.coordinates, z])
        inputs = torch.from_numpy(inputs)
        inputs = Variable(inputs, requires_grad=False, volatile=True)
        pixels = self.layers(inputs)
        if self.rgb:
            pixels = pixels.view(self.resolution[0], self.resolution[1], 3)
        else:
            pixels = pixels.view(self.resolution[0], self.resolution[1])
        pixels = pixels.data
        return pixels

    def image(self):
        z = self.random_z()
        image = self.forward(z)
        image = image.permute(2, 0, 1)
        image = tr.to_pil_image(image)
        return image

    def video(self, n_frames, z_speeds):
        pass

    def generate_coordinates(self):
        y = np.linspace(-self.z_scale, self.z_scale, self.resolution[0])
        x = np.linspace(-self.z_scale, self.z_scale, self.resolution[1])
        xx, yy = np.meshgrid(x, y)
        xx, yy = xx.flatten(), yy.flatten()
        coordinates = np.vstack([yy, xx]).transpose()
        r = np.linalg.norm(coordinates, axis=1)[:, np.newaxis]
        coordinates = np.hstack([coordinates, r])
        return coordinates.astype(np.float32)

    def random_z(self):
        z = np.random.uniform(-self.z_scale, self.z_scale, size=self.z_dim)
        z = np.reshape(np.tile(z, self.resolution[0] * self.resolution[1]),
                       [self.resolution[0] * self.resolution[1], self.z_dim])
        return z.astype(np.float32)

    def initialize_params(self, initialization):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if initialization['type'] == 'normal':
                    m.weight.data.normal_(std=initialization['stddev'])
                    if m.bias is not None:
                        m.bias.data.normal_(initialization['stddev'])
