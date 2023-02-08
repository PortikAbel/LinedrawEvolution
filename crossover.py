import numpy as np

from config import population_size

from initialize_population import MList
from selection import selection_types
from chromosome import Chromosome


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


"""
Crossover operators
"""


# def uniform_crossover(chromosome_1, chromosome_2, **kwargs):
#
#     index = np.random.randint(0, chromosome_1.origins.shape[0], 2)
#
#     print(f"crossover at {index}")
#
#     new_origins_1 = chromosome_1.origins.copy()
#     new_origins_2 = chromosome_2.origins.copy()
#
#     new_origins_1[index[0]] = chromosome_2.origins[index[1]]
#     new_origins_2[index[1]] = chromosome_1.origins[index[0]]
#
#     new_angles_1 = chromosome_1.angles.copy()
#     new_angles_2 = chromosome_2.angles.copy()
#
#     new_angles_1[index[0]] = chromosome_2.angles[index[1]]
#     new_angles_2[index[1]] = chromosome_1.angles[index[0]]
#
#     new_lengths_1 = chromosome_1.lengths.copy()
#     new_lengths_2 = chromosome_2.lengths.copy()
#
#     new_lengths_1[index[0]] = chromosome_2.lengths[index[1]]
#     new_lengths_2[index[1]] = chromosome_1.lengths[index[0]]
#
#     new_chromosome_1 = Chromosome(new_origins_1, new_angles_1, new_lengths_1, **kwargs)
#     new_chromosome_2 = Chromosome(new_origins_2, new_angles_2, new_lengths_2, **kwargs)
#
#     return new_chromosome_1, new_chromosome_2


def uniform_crossover(chromosome_1, chromosome_2, **kwargs):

    index = np.random.choice(chromosome_1.origins.shape[0], size=kwargs["num_lines_to_crossover"], replace=False)

    print(f"crossover at {index}")

    new_chromosome_1 = chromosome_1.copy()
    new_chromosome_2 = chromosome_2.copy()

    new_chromosome_1.angles[index] = chromosome_2.angles[index]
    new_chromosome_1.lengths[index] = chromosome_2.lengths[index]
    new_chromosome_1._fitness_scores[index] = chromosome_2._fitness_scores[index]
    new_chromosome_1._lines[index] = chromosome_2._lines[index]

    new_chromosome_2.angles[index] = chromosome_1.angles[index]
    new_chromosome_2.lengths[index] = chromosome_1.lengths[index]
    new_chromosome_2._fitness_scores[index] = chromosome_1._fitness_scores[index]
    new_chromosome_2._lines[index] = chromosome_1._lines[index]

    # print(f"parent 1 {chromosome_1.angles}, parent 2 {chromosome_2.angles}")
    # print(f"child 1 {new_chromosome_1.angles}, child 2 {new_chromosome_2.angles}")

    return new_chromosome_1, new_chromosome_2

def elitist_uniform_crossover(chromosome_1, chromosome_2, **kwargs):

    indices = np.random.choice(chromosome_1.origins.shape[0], size=kwargs["num_lines_to_crossover"], replace=False)

    # print(f"crossover at {indices}")

    new_chromosome_1 = chromosome_1.copy()
    new_chromosome_2 = chromosome_2.copy()

    for index in indices:

        chr = chromosome_1 if chromosome_1._fitness_scores[index] > chromosome_2._fitness_scores[index] else chromosome_2

        new_chromosome_1.angles[index] = chr.angles[index]
        new_chromosome_1.lengths[index] = chr.lengths[index]
        new_chromosome_1._fitness_scores[index] = chr._fitness_scores[index]
        new_chromosome_1._lines[index] = chr._lines[index]

        new_chromosome_2.angles[index] = chr.angles[index]
        new_chromosome_2.lengths[index] = chr.lengths[index]
        new_chromosome_2._fitness_scores[index] = chr._fitness_scores[index]
        new_chromosome_2._lines[index] = chr._lines[index]

    # print(f"parent 1 {chromosome_1.angles}, parent 2 {chromosome_2.angles}")
    # print(f"child 1 {new_chromosome_1.angles}, child 2 {new_chromosome_2.angles}")

    return new_chromosome_1, new_chromosome_2

crossover_types = {
    "uniform": uniform_crossover,
    "elitist_uniform": elitist_uniform_crossover,
}


def crossover_2(population, p, crossover_type, selection_type, **kwargs):
    """

    This function selects individuals and combines them using a crossover operator

    :param list population: The whole population
    :param float p: The probability of crossover happening to an individual
    :param string crossover_type: The type of the crossover i.e. uniform_crossover
    :param string selection_type: The type of the selection i.e random or elitist
    """

    subset = selection_types[selection_type](m=len(population), p=p, **kwargs)

    new_population = MList([])

    for i in range(0, len(subset), 2):
        ni = crossover_types[crossover_type](population[i], population[i+1], **kwargs)
        new_population.extend(ni)

    return new_population
