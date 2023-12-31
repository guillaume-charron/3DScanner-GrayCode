import cv2
import numpy as np

# Inspired from https://github.com/sjnarmstrong/gray-code-structured-light/blob/master/CaptureImages/GrayImages.py
def get_gray_codes(width, height):
    """
    Generates Gray codes for a given resolution.

    Parameters:
    ----------
    width : int
        The width of the image.
    height : int
        The height of the image.
    
    Returns:
    -------
    codes : np.array
        The Gray codes.
    """
    # Get larger size
    max_size = max(width, height)

    # Get length of gray codes to cover all pixels
    n_bits = int(np.ceil(np.log2(max_size)))

    # Generate Gray codes
    codes = np.arange(max_size, dtype=np.uint16)
    codes = (codes >> 1) ^ codes
    codes.byteswap(inplace=True)

    return np.unpackbits(codes.view(dtype=np.uint8)).reshape((-1, 16))[:, 16-n_bits:]

def get_image_sequence(gray_codes, width, height):
    """
    Generates a sequence of gray code images for a given resolution.

    Parameters:
    ----------
    gray_codes : np.array
        The Gray codes.
    width : int
        The width of the image.
    height : int
        The height of the image.
    
    Returns:
    -------
    images : np.array
        The Gray code images.
    """
    # Create black images
    images = np.zeros((4*len(gray_codes[0])+2,height, width), dtype=np.uint8)

    # Add all black and all white images
    images[0, :, :] = 0
    images[1, :, :] = 255 

    # Calculate the size each stripe
    stripe_size = width // (len(gray_codes))
    
    # Draw the Gray code pattern
    for i, code in enumerate(gray_codes):
        for j, bit in enumerate(code):
            start_pos = i * stripe_size
            end_pos = (i + 1) * stripe_size

            # Compute ids of frames, we want somthing like X9+,Y0+,X8+,Y1+,...,X9-,Y0-,X8-,Y1-,..
            # where + is normal and - is inverse
            id_v = 2*j + 2
            id_h = 2*(len(code) - (j+1)) + 3
            id_inv_v = 2*j + 2*len(code) + 2
            id_inv_h = 2*(len(code) - (j+1)) + 3 + 2*len(code)

            images[id_v, :, start_pos:end_pos] = 255 if bit == 1 else 0
            images[id_inv_v, :, start_pos:end_pos] = 255-images[id_v, :, start_pos:end_pos]
            if i < height:
                images[id_h, start_pos:end_pos, :] = 255 if bit == 1 else 0
                images[id_inv_h, start_pos:end_pos, :] = 255-images[id_h, start_pos:end_pos, :]
    
    return images

def display_gray_code(gray_codes, width, height, repeat_n, fps, write_video_seq=False):
    """
    Displays the Gray code pattern.
    
    Parameters:
    ----------
    gray_codes : np.array
        The Gray codes.
    width : int
        The width of the image.
    height : int
        The height of the image.
    repeat_n : int
        The number of times to repeat the sequence.
    fps : int
        The number of frames per second.
    write_video_seq : bool
        Whether to write the sequence to a video file.
    """
    images = get_image_sequence(gray_codes, width, height)

    if write_video_seq:
        out = cv2.VideoWriter('./data/gray_sequence.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), isColor=False)
    for n in range(repeat_n):
        for i in range(len(images)):
            # Display the image
            cv2.namedWindow('Gray Code Pattern', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('Gray Code Pattern', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Gray Code Pattern", images[i])
            cv2.waitKey(int(1000/fps))

            if write_video_seq and n == 0:
                out.write(images[i])
    
    cv2.destroyAllWindows()

    if write_video_seq:
        out.release()