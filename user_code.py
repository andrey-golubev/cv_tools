import numpy as np
import cv2

# output_buffer has an .add() method to write out a single user code line
def process_region(src_region, dst_region, output_buffer):
    mean = np.mean(dst_region, axis=(0, 1))
    if isinstance(mean, float):
        mean = [mean]
    mean = [int(e) for e in mean]
    output_buffer.add('Window mean:', mean)
    cv2.blur(dst_region, ksize=(11, 11), dst=dst_region)
    # cv2.Canny(dst_region, 100, 200, edges=dst_region)

# return dst_image even if it is unchanged
def process_image(src_image, dst_image, output_buffer):
    dst_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    return dst_image

def preprocess(src_image):
    # cv2.blur(src_image, ksize=(21, 21), dst=src_image)
    return src_image
