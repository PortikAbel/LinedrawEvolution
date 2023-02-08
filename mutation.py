import numpy as np

from util import gen_random_angles
from selection import selection_types
from initialize_population import MList

from fitness import error_function_2

def mutation_operator_1(individual):

    m = individual.shape[0]

    i = np.random.randint(m, size=m//10)

    individual[i] += (np.random.rand(m//10, 2, 2) * 10).astype(int)

    individual[individual > 1180] = 1180
    individual[individual < 0] = 0

    return individual


def mutation_operator_2(individual, target):

    scores = error_function_2(individual, target)

    scores -= min(scores)
    scores = scores.astype(np.float64)
    scores /= sum(scores)
    
    num_lines = individual.shape[0]
    line_to_mutate = np.random.choice(num_lines, p=scores)

    indices = np.random.choice(target.shape[0], 2, replace=False)

    individual[line_to_mutate] = target[indices]



def mutation(population, target, p=.5, operator=mutation_operator_2):
    """
    This function selects some individuals from the population and mutates them

    :param list population: The whole population
    :param float p: The probability of mutation happening to an individual
    :param func operator: The mutation operator
    """
    probabilities = np.random.random(len(population))
    subset = probabilities > p

    for individual in population[subset]:
        operator(individual, target)

    return subset


"""
Mutation operators
"""


def non_uniform_mutation(chromosome, **kwargs):
    """
    This function modifies either all angles or all lengths
    """

    num_lines = kwargs["num_lines_to_mutate"]

    assert chromosome.angles.shape[0] >= num_lines > 0

    p = 1 - chromosome._fitness_scores
    p /= sum(p)

    indices = np.random.choice(range(chromosome.angles.shape[0]),
                               num_lines,
                               replace=False,
                               p=p)

    new_chromosome = chromosome

    prob = np.random.rand()

    p = chromosome._fitness_scores
    p /= sum(p)

    if prob < p[indices][0] / 1 / len(p):

        # offset angles by 5 degrees, then clip to [0, pi)
        angles_sample = np.random.choice([-1, 1], num_lines) * 0.00872665
        # new_chromosome.angles[indices] = np.clip(chromosome.angles[indices] + angles_sample, 0, np.pi - 1.e-6)
        new_chromosome.angles[indices] = chromosome.angles[indices] + angles_sample

    else:

        angles_sample = np.random.choice([-1, 1], num_lines) * 1.57
        # new_chromosome.angles[indices] = np.clip(chromosome.angles[indices] + angles_sample, 0, np.pi - 1.e-6)
        new_chromosome.angles[indices] = chromosome.angles[indices] + angles_sample

        # # slightly increase or decrease line length
        # lengths_sample = np.random.randint(-10, 11, num_lines)
        # chromosome.lengths[indices] += lengths_sample

    new_chromosome._modified[indices] = True

    # print(f"Modified lines {indices}")

    return new_chromosome


def uniform_mutation(chromosome, **kwargs):
    """
    This function randomly resets either all angles or all lengths
    """

    num_lines = chromosome.angles.shape[0]

    prob = np.random.rand()

    if prob < .5:

        angles_sample = gen_random_angles(num_lines)
        chromosome.angles = angles_sample

    else:

        lengths_sample = np.random.randint(10, kwargs["in_line_length"], num_lines)
        chromosome.lengths = lengths_sample

    return chromosome


mutation_types = {
    "non_uniform": non_uniform_mutation,
    "uniform": uniform_mutation,
}

"""
Function to perform selection and mutation on population
"""


def mutation_2(population, p, mutation_type, selection_type, **kwargs):
    """
    This function selects some individuals from the population and mutates them

    :param list population: The whole population
    :param float p: The probability of mutation happening to an individual
    :param string mutation_type: The type of mutation
    :param string selection_type: The type of selection
    :param ndarray sort_index: The indices sorting the population
    """
    subset = selection_types[selection_type](m=len(population), p=p, **kwargs)

    mutated_population = MList([mutation_types[mutation_type](xi, **kwargs) for xi in population[subset]])

    return mutated_population
