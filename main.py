import cv2
import numpy as np

from genetic_alg import run_genetic_algorithm
from config import (
    population_size,
    num_epochs,
    line_count,
    target_image_path,
    result_image_path,
)


if __name__=="__main__":
    img = cv2.imread(target_image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # res = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
    target = cv2.Canny(gray, 100, 200)

    current = np.zeros_like(target, dtype=np.uint8)
    pop = run_genetic_algorithm(population_size, num_epochs, line_count, target)
    img = cv2.polylines(current, pop[0], False, 255)
    
    cv2.imwrite(result_image_path, img)