import numpy as np

from config import population_size


def crossover_operator_1(individuals):
    individual_1, individual_2 = individuals
    cross_point = np.random.randint(0, individual_1.shape[0])
    return np.array((
        np.vstack((individual_1[:cross_point], individual_2[cross_point:])), 
        np.vstack((individual_2[:cross_point], individual_1[cross_point:]))
    ))


def crossover(population, p=.5, operator=crossover_operator_1):
    """

    This function selects individuals and combines them using a crossover operator

    :param list population: The whole population
    :param float p: The probability of crossover happening to an individual
    :param func operator: The crossover function
    :param func selector: Selection function to use if specified 
                          i.e. fitness based selector,
                          default is random selection

    """

    probabilities = np.random.random(population_size // 2)
    subset = probabilities > p
    pairs = np.random.choice(population_size, (population_size // 2, 2), replace=False)
    pairs = pairs[subset]

    left_operands = population[pairs[:, 0]]
    right_operands = population[pairs[:, 1]]

    return np.concatenate(list(map(operator, zip(left_operands, right_operands))))
