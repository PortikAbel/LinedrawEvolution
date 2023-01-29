import os

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
from util import black_white_img_to_line_draw


if __name__=="__main__":

    target = black_white_img_to_line_draw(
        os.path.join(input_dir, image_name),
        image_shape
    )

    log_dir = os.path.join(
        output_dir,
        image_name.split(".")[0],
    )

    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "error.csv"), "w") as error_log:

        pop, best_individuals = run_genetic_algorithm(
            population_size,
            num_epochs,
            line_count,
            target,
            log_dir,
            error_log,
        )