import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from open3d import core as o3c

def arr_creat(upperleft, upperright, lowerleft, lowerright, arrwidth, arrheight):
    arr = np.linspace(np.linspace(lowerleft, lowerright, arrwidth), 
                      np.linspace(upperleft, upperright, arrwidth), arrheight, dtype=int)
    return arr[:, :, None]

def create_color_map(width, height):
    r = arr_creat(0,   255, 0,   255, width, height)
    g = arr_creat(0,   0,   255, 0, width, height)
    b = arr_creat(255, 255, 0,   0, width, height)

    img = np.concatenate([r, g, b], axis=2)

    return img

def plot_decoded_graycodes(h_pixels, v_pixels, color_map_img):
    width = color_map_img.shape[1]
    height = color_map_img.shape[0]

    result_img = np.zeros((h_pixels.shape[0], h_pixels.shape[1], 3), dtype=int)

    for i in range(h_pixels.shape[1]):
        for j in range(h_pixels.shape[0]):
            h_value = h_pixels[j, i]
            v_value = v_pixels[j, i]
            if h_value == -1 or v_value == -1:
                result_img[j, i] = [0, 0, 0]
            else:
                h_value = min(width-1, h_value)
                v_value = min(height-1, v_value)
                result_img[j, i] = color_map_img[v_value, h_value]

    plt.imshow(result_img)
    plt.axis('off')
    plt.show()

def plot_point_cloud(Pts, colors, save_to):
    pcd = o3d.t.geometry.PointCloud(o3c.Tensor(Pts.T, o3c.float32))
    pcd = pcd.to_legacy()
    pcd.colors = o3d.cpu.pybind.utility.Vector3dVector(colors)

    print('Remove outlier in point cloud')
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
    inlier_cloud = pcd.select_by_index(ind)

    print('Save ply file')
    o3d.io.write_point_cloud(os.path.join(save_to, 'cloud.ply'), inlier_cloud)

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        id = 0
        while os.path.exists(os.path.join(save_to, f'cloud_{id}.png')):
            id+=1
        plt.imsave(os.path.join(save_to,f'cloud_{id}.png'), np.asarray(image))
        return False

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord(".")] = capture_image
    o3d.visualization.draw_geometries_with_key_callbacks([inlier_cloud], key_to_callback)
