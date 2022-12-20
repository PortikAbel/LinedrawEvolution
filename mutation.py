import numpy as np

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
