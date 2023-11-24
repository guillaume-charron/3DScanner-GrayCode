import cv2
import numpy as np

# Inspired from https://github.com/sjnarmstrong/gray-code-structured-light/blob/master/CaptureImages/GrayImages.py
def get_gray_codes(width, height):
    # Get larger size
    max_size = max(width, height)

    # Get length of gray codes to cover all pixels
    n_bits = int(np.ceil(np.log2(max_size)))

    # Generate Gray codes
    codes = np.arange(max_size, dtype=np.uint16)
    codes = (codes >> 1) ^ codes
    codes.byteswap(inplace=True)

    return np.unpackbits(codes.view(dtype=np.uint8)).reshape((-1, 16))[:, 16-n_bits:]

def display_gray_code(gray_codes, width, height, repeat_n, fps, write_video_seq=False):
    # Create a black image
    image = np.zeros((4*len(gray_codes[0]),height, width), dtype=np.uint8)

    # Calculate the size each stripe
    stripe_size = width // (len(gray_codes))
    
    # Draw the Gray code pattern
    for i, code in enumerate(gray_codes):
        for j, bit in enumerate(code):
            start_pos = i * stripe_size
            end_pos = (i + 1) * stripe_size

            # Compute ids of frames, we want somthing like X9+,Y0+,X8+,Y1+,...,X9-,Y0-,X8-,Y1-,..
            # where + is normal and - is inverse
            id_v = 2*j
            id_h = 2*(len(code) - (j+1)) + 1
            id_inv_v = 2*j + 2*len(code)
            id_inv_h = 2*(len(code) - (j+1)) + 1 + 2*len(code)

            image[id_v, :, start_pos:end_pos] = 255 if bit == 1 else 0
            image[id_inv_v, :, start_pos:end_pos] = 255-image[id_v, :, start_pos:end_pos]
            if i < height - 1:
                image[id_h, start_pos:end_pos, :] = 255 if bit == 1 else 0
                image[id_inv_h, start_pos:end_pos, :] = 255-image[id_h, start_pos:end_pos, :]
    
    if write_video_seq:
        out = cv2.VideoWriter('./data/gray_sequence.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), isColor=False)
 
    for n in range(repeat_n):
        for i in range(len(image)):
            # Display the image
            cv2.imshow("Gray Code Pattern", image[i])
            cv2.waitKey(int(1000/fps))

            if write_video_seq and n == 0:
                out.write(image[i])
    
    cv2.destroyAllWindows()

    if write_video_seq:
        out.release()


# Get gray codes
width, height = (1920, 1080)
gray_codes = get_gray_codes(width, height)

# Display gray codes
repeat_n = 1
fps = 10
display_gray_code(gray_codes, width, height, repeat_n, fps, write_video_seq=True)