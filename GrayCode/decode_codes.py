import os
import cv2
import numpy as np

def read_images(folder):
    im = []
    for file_name in sorted(os.listdir(folder), key=len):
        im.append((file_name,cv2.imread(os.path.join(folder, file_name))))
    return im

def remove_bad_images(images):
    diff1 = len(np.argwhere(cv2.absdiff(images[0][1], images[1][1]) > 50))
    diff2 = len(np.argwhere(cv2.absdiff(images[1][1], images[2][1]) > 50))
    filtered_indexes = []
    for i, (file_name, image) in enumerate(images[2:-2]):
        sub = cv2.absdiff(images[i+3][1], image)
        diff3 = len(np.argwhere(sub > 50))
        print(diff1,diff2,diff3)
        

        if diff1 < diff2 and diff1 < diff3 and diff2 < diff3 and i+1 not in filtered_indexes and diff1 < 100000:
            filtered_indexes.append(i+1)
        elif diff2 <= diff1 and diff2 <= diff3 and i+3 not in filtered_indexes and diff2 < 100000:
            filtered_indexes.append(i+3)
        diff1 = diff2
        diff2 = diff3

    return filtered_indexes


images = read_images('./data/recordings/record_1')
filtered_indexes = remove_bad_images(images)
for i in filtered_indexes:
    print(images[i][0])
print(len(images), len(filtered_indexes))