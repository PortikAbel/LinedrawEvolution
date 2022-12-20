import numpy as np
import cv2
from skimage.draw import line as discrete_line


def error_function_1(individual, target):

    current = np.zeros_like(target)
    cv2.polylines(current, individual, False, 255)
    difference = target - current

    return cv2.countNonZero(difference)


def error_function_2(individual, target):

    positive_scores = []
    negative_scores = []

    target_as_list = list(map(tuple, target))

    for line in individual:
        points = list(zip(*discrete_line(*line[0], *line[1])))

        line_intersect_target = set(points).intersection(target_as_list)
        line_minus_target = set(points).difference(target_as_list)

        positive_scores.append(len(line_intersect_target))
        negative_scores.append(len(line_minus_target))

    return np.array(negative_scores) - np.array(positive_scores)


def fitness(population, target, error_function=error_function_2):

    error = np.array(list(map(
        lambda individual: sum(error_function(individual, target)), population
    )))
    sort_index = np.argsort(error)

    return error, sort_index
