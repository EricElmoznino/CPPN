import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import functional as tr
import numpy as np
import math
import cv2
from tqdm import tqdm


class CPPN(nn.Module):
    def __init__(self, resolution, rgb, z_dim, z_scale, n_layers, layer_sizes, activations, initialization):
        super().__init__()
        self.eval()

        if type(layer_sizes) == list:
            assert n_layers == len(layer_sizes)
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
        inputs = Variable(inputs, requires_grad=False)
        with torch.no_grad():
            pixels = self.layers(inputs)
            pixels = pixels.view(self.resolution[0], self.resolution[1], 3 if self.rgb else 1)
            pixels = pixels.data
        return pixels

    def image(self, save_path=None, z=None):
        if z is None:
            z = self.random_z()
        image = self.forward(z)
        image = image.permute(2, 0, 1)
        image = tr.to_pil_image(image)
        if save_path is not None:
            image.save(save_path)
        return image

    def video(self, save_path, length, z_cycles):
        assert len(z_cycles) == self.z_dim
        n_frames = int(length * 60)
        z_cycles = np.array(z_cycles).astype(np.float32)
        z_periods = length / z_cycles

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_writer = cv2.VideoWriter(save_path, fourcc, 60, (self.resolution[1], self.resolution[0]))

        for t in tqdm(range(n_frames)):
            z = np.sin(2*math.pi/z_periods * t/n_frames * length) * self.z_scale
            z = self.tile_z(z)
            frame = self.image(z=z).convert('RGB')
            video_writer.write(np.array(frame))

        video_writer.release()

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
        z = self.tile_z(z)
        return z.astype(np.float32)

    def tile_z(self, z):
        return np.reshape(np.tile(z, self.resolution[0] * self.resolution[1]),
                          [self.resolution[0] * self.resolution[1], self.z_dim])

    def initialize_params(self, initialization):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if initialization['type'] == 'normal':
                    m.weight.data.normal_(std=initialization['stddev'])
                    if m.bias is not None:
                        m.bias.data.normal_(initialization['stddev'])
