import numpy as np
import cv2

def black_white_img_to_line_draw(path, image_shape):
    img = cv2.imread(path)
    img = cv2.resize(img, image_shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 100, 200)

def target_img_to_sparse(target_img):
    white_y, white_x = np.nonzero(target_img)
    return np.array(list(zip(white_x, white_y)))
