import numpy as np
import matplotlib.pyplot as plt

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
    plt.show()