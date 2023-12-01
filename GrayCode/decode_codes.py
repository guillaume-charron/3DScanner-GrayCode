import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

def read_images(folder):
    files = sorted(os.listdir(folder), key=len)
    images = None
    file_names = np.array([])
    for i, file_name in enumerate(tqdm(files)):
        im = cv2.imread(os.path.join(folder, file_name))
        if images is None:
            images = np.empty((len(files), im.shape[0], im.shape[1], im.shape[2]))

        file_names = np.append(file_names, file_name)
        images[i] = im
    return images, file_names

def remove_bad_images(images):
    diff_thresh = 50
    diff1 = len(np.argwhere(cv2.absdiff(images[0], images[1]) > diff_thresh))
    diff2 = len(np.argwhere(cv2.absdiff(images[1], images[2]) > diff_thresh))
    filtered_indexes = []
    for i, image in enumerate(tqdm(images[2:-1])):
        sub = cv2.absdiff(images[i+3], image)
        diff3 = len(np.argwhere(sub > diff_thresh))
        #print(diff1,diff2,diff3)

        if diff1 < diff2 and diff1 < diff3 and diff2 < diff3 and i+1 not in filtered_indexes:
            if len(filtered_indexes) == 0 or (filtered_indexes[len(filtered_indexes)-1] != i and filtered_indexes[len(filtered_indexes)-1] != i+2):
                filtered_indexes.append(i+1)
        elif diff2 <= diff1 and diff2 <= diff3 and i+2 not in filtered_indexes:
            if len(filtered_indexes) == 0 or (filtered_indexes[len(filtered_indexes)-1] != i+1 and filtered_indexes[len(filtered_indexes)-1] != i+3):
                filtered_indexes.append(i+2)
                diff3 = -1
                diff2 = -1

        diff1 = diff2
        diff2 = diff3

    return filtered_indexes

def decode_images(images):
    per_pixel_thresh = (images[1] - images[0]) / 2 + images[0]



if __name__ == '__main__':
    folder = './data/recordings/record_1'
    filtered_folder = './data/recordings/record_1_filtered'
    images = None
    if not os.path.exists(filtered_folder):
        os.mkdir(filtered_folder)

        print('Read images')
        images, file_names = read_images(folder)
        print('Remove bad images')
        filtered_indexes = remove_bad_images(images)
        print('Save filtered images')
        for file_name in tqdm(file_names[filtered_indexes]):
            shutil.copy(os.path.join(folder,file_name), os.path.join(filtered_folder,file_name))
        images = images[filtered_indexes]
    else:
        print('Read images')
        images, file_names = read_images(filtered_folder)


    print(f'Found {len(images)} good images')