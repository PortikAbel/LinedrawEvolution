from config import population_size
import numpy as np

def elitist(upspring, error):
    sort_idx = error.argsort()
    return upspring[sort_idx[:population_size]]

def selection(upspring, error):
    return elitist(upspring, error)


def select_next_generation(population, n):
    scores = np.asarray([chromosome.fitness() for chromosome in population])

    return np.argsort(scores)[:n]


def select_random(**kwargs):
    """
    Returns an array of (m * p) random indices
    m: population size
    p: probablity of sampling
    """

    return np.random.choice(range(kwargs['m']), size=int(kwargs['m'] * kwargs['p']), replace=False)


def select_elite(**kwargs):
    """
    Returns an array of the first (m * p) or (m * p - 1) indices
    m: population size
    p: probability of sampling
    sort_index: indices sorted by fitness
    """

    m = int(kwargs['m'] * kwargs['p']) \
        if int(kwargs['m'] * kwargs['p']) % 2 == 0 else \
        int(kwargs['m'] * kwargs['p']) - 1

    return kwargs['sort_index'][:m]


selection_types = {
    "random": select_random,
    "elite": select_elite,
}