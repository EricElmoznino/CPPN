import time

from CPPN import CPPN
import config

cppn = CPPN(config.resolution, config.rgb, config.z_dim, config.z_scale, config.n_layers,
            config.layer_sizes, config.activations, config.initialization)

# _ = cppn.image(save_path='generated_images/test/' + str(time.time()) + '.png')
cppn.video(save_path='generated_videos/test/' + str(time.time()) + '.mp4', length=config.video_length, z_cycles=config.video_z_cycles)
