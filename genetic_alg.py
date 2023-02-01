import os
import sys

import cv2
import numpy as np

from chromosome import Chromosome
from initialize_population import init_population, initialize_population, MList
from mutation import mutation, mutation_2
from crossover import crossover, crossover_2
from fitness import fitness
from selection import selection, select_next_generation
from util import target_img_to_sparse
import config

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


def run_genetic_algorithm_2(n, N, m, target_img, log_dir, error_log=sys.stdout.buffer, epsilon=1e-2):
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

    target_img = np.pad(target_img, 100, mode='constant')

    best_individuals = []
    target_sparse = target_img_to_sparse(target_img)

    in_line_len = (target_img.shape[0] + target_img.shape[1]) / 2 / 30
    population = initialize_population(n, m, target_sparse, in_line_len, target_img)

    i = 0
    while i < N:
        i += 1
        print(i)
        generation_dir = os.path.join(log_dir, f"gen_{i}")
        os.makedirs(generation_dir, exist_ok=True)

        print('crossover_select')

        sorted_pop_ind = select_next_generation(population, n)

        children = crossover_2(population, .8,
                               crossover_type=config.crossover_type,
                               selection_type=config.selection_type,
                               target=target_img,
                               sort_index=sorted_pop_ind)

        new_population = MList(population + children)

        print('mutation_select')

        sorted_pop_ind = select_next_generation(new_population, len(new_population))

        _ = mutation_2(new_population, .05,
                       mutation_type=config.mutation_type,
                       selection_type=config.selection_type,
                       in_line_length=in_line_len,
                       sort_index=sorted_pop_ind)

        print('new_pop_select')

        sorted_pop_ind = select_next_generation(new_population, n)

        population = new_population[sorted_pop_ind]

        best_individuals.append(population[0])

        pts = Chromosome.genes_to_lines(population[0].origins, population[0].angles, population[0].lengths)

        nimg = np.zeros(target_img.shape)

        cv2.polylines(nimg, pts, False, 255)

        cv2.imshow("best individual", nimg)

        cv2.waitKey(0)

        # np.save(
        #     os.path.join(generation_dir, f"offspring_gen_{i}.npy"),
        #     up,
        # )

        # error = fitness(up, target_img)
        # np.savetxt(
        #     error_log,
        #     error.reshape(1, error.shape[0]),
        #     delimiter=";",
        #     fmt="%.10f",
        # )
        # error_log.flush()
        #
        # best_idx = error.argmin()
        # np.save(
        #     os.path.join(generation_dir, f"best_gen_{i}.npy"),
        #     up[best_idx],
        # )
        # best_individuals.append(up[best_idx].copy())
        #
        # if error[best_idx] < epsilon:
        #     break
        # population = selection(up, error)
        # np.save(
        #     os.path.join(generation_dir, f"selection_gen_{i}.npy"),
        #     population,
        # )

    return population, best_individuals