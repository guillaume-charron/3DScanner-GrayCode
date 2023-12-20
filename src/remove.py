from matplotlib import pyplot as plt
import numpy as np
from scanner.utils import visualize

# proj_width, proj_height = (1280, 720)

# record_name = 'bouda_720'
# h_pixels = np.load(f'./data/recordings/{record_name}/h_pixels.npy')
# v_pixels = np.load(f'./data/recordings/{record_name}/v_pixels.npy')

# # Display decoded gray codes
# color_map_img = visualize.create_color_map(proj_width, proj_height)
# plt.axis('off')
# plt.imshow(color_map_img)
# plt.show()
# #visualize.plot_decoded_graycodes(h_pixels, v_pixels, color_map_img)

cam_mtx = './data/calib_results/proj/proj_mtx.npy'
print(np.load(cam_mtx))
