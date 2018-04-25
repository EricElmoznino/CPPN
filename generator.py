import time

from CPPN import CPPN
import config

cppn = CPPN(config.resolution, config.rgb, config.z_dim, config.z_scale, config.n_layers,
            config.layer_sizes, config.activations, config.initialization)
image = cppn.image()
image.save('generated_images/test/' + str(time.time()) + '.png')
