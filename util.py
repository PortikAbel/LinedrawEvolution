import numpy as np


def target_img_to_sparse(target_img):
    white_y, white_x = np.nonzero(target_img)
    return np.array(list(zip(white_x, white_y)))