import cv2
import numpy as np

# took from https://www.geeksforgeeks.org/generate-n-bit-gray-codes/
def get_gray_codes(n):
    codes = []

    for i in range(1 << n):
        val = (i ^ (i >> 1))
        s = bin(val)[2::]

        codes.append(s.zfill(n))
    return codes

def display_gray_code(gray_codes, width, height, repeat_n, fps):
    # Create a black image
    image = np.zeros((4*len(gray_codes[0]),height, width), dtype=np.uint8)

    # Calculate the size each stripe
    stripe_width = width / (len(gray_codes))
    stripe_height = height / (len(gray_codes))
    
    # Draw the Gray code pattern
    for i, code in enumerate(gray_codes):
        for j, bit in enumerate(code):
            start_col = int(np.round(i * stripe_width))
            end_col = int(np.round((i + 1) * stripe_width))
            start_row = int(np.round(i * stripe_height))
            end_row = int(np.round((i + 1) * stripe_height))

            if i == len(gray_codes) - 1:
                end_col = width
                end_row = height
            image[4*j, :, start_col:end_col] = 255 if bit == '1' else 0
            image[4*j+1, :, start_col:end_col] = 255-image[4*j, :, start_col:end_col]
        
            image[4*(len(code) - (j+1)) + 2, start_row:end_row, :] = 255 if bit == '1' else 0
            image[4*(len(code) - (j+1)) + 3, start_row:end_row, :] = 255-image[4*(len(code) - (j+1)) + 2, start_row:end_row, :]
    
    #out = cv2.VideoWriter('gray_sequence.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height), isColor=False)
 
    for n in range(repeat_n):
        for i in range(len(image)):
        # Display the image
        #out.write(image[i])
        
            cv2.imshow("Gray Code Pattern", image[i])
            cv2.waitKey(int(1000/fps))
    cv2.destroyAllWindows()
    #out.release()


# Get gray codes
gray_codes_width = get_gray_codes(9)

# Display gray codes
width, height = (1920, 1080)
repeat_n = 5
fps = 10
display_gray_code(gray_codes_width, width, height, repeat_n, fps)