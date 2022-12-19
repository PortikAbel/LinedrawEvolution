import numpy as np

from initialize_population import init_population
from mutation import mutation
from crossover import crossover
from fitness import fitness


def run_genetic_algorithm(n, N, m, target, delta=10, epsilon=100):
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

    best_fitness = []
    population = np.asarray(init_population(
        n, num_lines=m, img_shape=target.shape, max_line_length=10
    ))

    i = 0
    while i < N:
        i += 1
        print(i)

        cp = crossover(population)
        up = np.concatenate((population, cp), axis=0)

        _ = mutation(up)

        error, sort_index = fitness(up, target)

        print(error)

        best = error[sort_index[0]]
        best_fitness.append(best)

        if best < epsilon:
            break
        population = up[sort_index][:n]

    return population
