import os
import argparse
import numpy as np
import cv2

from config import (
    input_dir,
    output_dir,
    image_name,
    image_shape,
)
from util import (
    black_white_img_to_line_draw,
    merge_line_draws,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--mode",
    required=True, nargs=1, type=str,
    choices=["b", "best", "o", "offspring", "s", "selection"],
)
parser.add_argument(
    "-g", "--generation",
    nargs=1, type=int, default=1,
)

args = parser.parse_args()

mode = args.mode[0]
generation = args.generation[0]

target = black_white_img_to_line_draw(
    os.path.join(input_dir, image_name),
    image_shape
)
empty_img = np.zeros_like(target, dtype=np.uint8)

img_dir = os.path.join(output_dir, "img")
os.makedirs(img_dir, exist_ok=True)

if mode in ["b", "best"]:
    best_path = os.path.join(
        output_dir,
        image_name.split(".")[0],
        f"gen_{generation}",
        f"best_gen_{generation}.npy",
    )
    img_best_path = os.path.join(
        img_dir,
        f"best_{image_name}",
    )

    best = np.load(best_path)
    img = cv2.polylines(empty_img, best, False, 255)
    cv2.imwrite(img_best_path, merge_line_draws(target, img))

if mode in ["o", "offspring"]:
    offspring_path = os.path.join(
        output_dir,
        image_name.split(".")[0],
        f"gen_{generation}",
        f"offspring_gen_{generation}.npy",
    )
    offspring = np.load(offspring_path)

    for i, individual in enumerate(offspring):
        img_offspring_path = os.path.join(
            img_dir,
            f"offspring_{i}_{image_name}",
        )
        img = cv2.polylines(empty_img, individual, False, 255)
        cv2.imwrite(os.path.join(output_dir, f"{i}_{image_name}"), merge_line_draws(target, img))

if mode in ["s", "selection"]:
    selection_path = os.path.join(
        output_dir,
        image_name.split(".")[0],
        f"gen_{generation}",
        f"selection_gen_{generation}.npy",
    )
    selection = np.load(selection_path)

    for i, individual in enumerate(selection):
        img_selection_path = os.path.join(
            img_dir,
            f"selection_{i}_{image_name}",
        )
        img = cv2.polylines(empty_img, individual, False, 255)
        cv2.imwrite(os.path.join(output_dir, f"{i}_{image_name}"), merge_line_draws(target, img))