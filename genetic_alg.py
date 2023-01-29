import os
import sys
import numpy as np

from initialize_population import init_population
from mutation import mutation
from crossover import crossover
from fitness import fitness
from selection import selection
from util import target_img_to_sparse


def run_genetic_algorithm(n, N, m, target_img, log_dir, error_log=sys.stdout.buffer, epsilon=1e-2):
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
    population = init_population(n, num_lines=m, target_sparse=target_sparse)

    i = 0
    while i < N:
        i += 1
        generation_dir = os.path.join(log_dir, f"gen_{i}")
        os.makedirs(generation_dir, exist_ok=True)

        cross_population = crossover(population, .8)
        up = np.concatenate((population, cross_population), axis=0)

        _ = mutation(up, target_sparse, .05)
        np.save(
            os.path.join(generation_dir, f"offspring_gen_{i}.npy"),
            up,
        )

        error = fitness(up, target_img)
        np.savetxt(
            error_log,
            error.reshape(1, error.shape[0]),
            delimiter=";",
            fmt="%.10f",
        )
        error_log.flush()

        best_idx = error.argmin()
        np.save(
            os.path.join(generation_dir, f"best_gen_{i}.npy"),
            up[best_idx],
        )
        best_individuals.append(up[best_idx].copy())

        if error[best_idx] < epsilon:
            break
        population = selection(up, error)
        np.save(
            os.path.join(generation_dir, f"selection_gen_{i}.npy"),
            population,
        )


    return population, best_individuals
