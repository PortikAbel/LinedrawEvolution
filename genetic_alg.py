import numpy as np

from initialize_population import init_population
from mutation import mutation
from crossover import crossover
from fitness import fitness
from util import target_img_to_sparse


def run_genetic_algorithm(n, N, m, target_img, delta=10, epsilon=1e-2):
    """
    This function runs a genetic algorithm until N iterations is reached
    or best fitness value is smaller than epsilon

    Parameters
    ----------
    :param int n: The size of the population
    :param int N: The number of iterations
    :param int m: The number of lines that constitute an image
    :param ndarray target: The target image
    :param float delta: Number of iterations without progress. Stop condition
    :param float epsilon: Lower bound for fitness value. Stop condition

    """

    best_individuals = []
    target_sparse = target_img_to_sparse(target_img)
    population = np.asarray(init_population(n, num_lines=m, target_sparse=target_sparse))

    i = 0
    while i < N:
        i += 1
        print(f"generation {i}")

        cross_population = crossover(population, .8)
        up = np.concatenate((population, cross_population), axis=0)

        _ = mutation(up, target_sparse, .05)

        error, sort_index = fitness(up, target_img)

        print(error)

        best = error[sort_index[0]]
        best_individuals.append(up[sort_index[0]].copy())

        if best < epsilon:
            break
        population = up[sort_index[:n]]

    return population, best_individuals
