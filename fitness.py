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

def error_function_3(individual, target):
    """
    The difference of an individual by
    """
    scores = []
    white = target.max()

    for line in individual:

        start_row, end_row = min(line[0][0], line[1][0]), max(line[0][0], line[1][0]) + 1
        start_col, end_col = min(line[0][1], line[1][1]), max(line[0][1], line[1][1]) + 1
        row_count = end_row - start_row
        col_count = end_col - start_col

        line_offset = line - [start_row, start_col]

        line_in_matrix = np.zeros((row_count, col_count))
        line_in_matrix[discrete_line(
            *line_offset.reshape(-1)
        )] = white

        # target_crop = np.zeros_like(line_in_matrix)
        target_crop = target[start_row:end_row, start_col:end_col]

        svd_1 = np.linalg.svd(line_in_matrix)
        svd_1 = np.concatenate((svd_1[0].reshape(-1), svd_1[2].reshape(-1)))
        svd_2 = np.linalg.svd(target_crop)
        svd_2 = np.concatenate((svd_2[0].reshape(-1), svd_2[2].reshape(-1)))

        scores.append(np.average(np.abs(np.subtract(svd_1, svd_2))))

    return scores

def fitness(population, target, error_function=error_function_3):
    """
    Calculating the fitness of each individual in a population

    Parameters
    ----------
    :param numpy.ndarray population: The array of coordinates of endpoints of lines of an individual
    :param numpy.ndarray target: The target matrix approximated by discrete lines
    """
    error = np.array(list(map(
        lambda individual: sum(error_function(individual, target)), population
    )))
    sort_index = np.argsort(error)

    return error, sort_index
