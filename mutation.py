import numpy as np

def mutation_operator_1(individual):
  
  m = individual.shape[0]

  i = np.random.randint(m, size=m//10)

  individual[i] += (np.random.rand(m//10,2,2) * 10).astype(int)

  individual[individual > 1180] = 1180
  individual[individual < 0] = 0

  return individual


def mutation(population, p=.5, operator=mutation_operator_1):
  """
  This function selects some individuals from the population and mutates them

  :param list population: The whole population
  :param float p: The probability of mutation happening to an individual
  :param func operator: The mutation operator
  """
  subset = np.random.choice(range(len(population)), 
                            size=int(len(population) * p),
                            replace=False)

  population[subset] = np.asarray([operator(xi) for xi in population[subset]]) 
  
  return subset
