import numpy as np
import cv2

def error_function_1(individual, target):

  current = np.zeros_like(target)

  cv2.polylines(current, individual, False, 255)

  difference = target - current

  return cv2.countNonZero(difference)


def error_func_2(individual, target):

  positive_scores = []
  negative_scores = []

  for line in individual:

    points = list(zip(*discrete_line(*line[0], *line[1])))

    line_intersect_target = set(points).intersection(target)

    line_minus_target = set(points).difference(target)

    positive_scores.append(len(line_intersect_target))

    negative_scores.append(len(line_minus_target))

  return positive_scores, negative_scores


def fitness(population, target, error_function=error_function_1):
  
  error = np.asarray([error_function(individual, target) 
                     for individual in population])
  
  sort_index = np.argsort(error)

  return error, sort_index