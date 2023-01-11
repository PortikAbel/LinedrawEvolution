import os
import cv2
import numpy as np

from genetic_alg import run_genetic_algorithm
from config import (
    population_size,
    num_epochs,
    line_count,
    image_shape,
    input_dir,
    output_dir,
    image_name,
)


if __name__=="__main__":
    img = cv2.imread(os.path.join(input_dir, image_name))
    img = cv2.resize(img, image_shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # res = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
    target = cv2.Canny(gray, 100, 200)

    current = np.zeros_like(target, dtype=np.uint8)
    pop, best_individuals = run_genetic_algorithm(population_size, num_epochs, line_count, target)
    img = cv2.polylines(current, pop[0], False, 255)
    
    cv2.imwrite(os.path.join(output_dir, image_name), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) * (0, 255, 0) + cv2.cvtColor(target, cv2.COLOR_GRAY2BGR) * (255, 0, 0))
    for i, individual in enumerate(best_individuals):
        img = cv2.polylines(current, individual, False, 255)
        cv2.imwrite(os.path.join(output_dir, f"{i}_{image_name}"), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) * (0, 255, 0) + cv2.cvtColor(target, cv2.COLOR_GRAY2BGR) * (255, 0, 0))