import numpy as np

def crossover_operator_1(individual_1, individual_2):
    
  i1_1, i1_2 = np.split(individual_1, 2)

  i2_1, i2_2 = np.split(individual_2, 2)

  return np.vstack((i1_1, i2_2)), np.vstack((i2_1, i1_2))

def crossover(population, p=.5, operator=crossover_operator_1, selector=None):
  """

  This function selects individuals and combines them using a crossover operator

  :param list population: The whole population
  :param float p: The probability of crossover happening to an individual
  :param func operator: The crossover function
  :param func selector: Selection function to use if specified 
                        i.e. fitness based selector,
                        default is random selection

  """

  if selector is not None:
    index = selector(population)
  else:
    index = np.random.choice(range(len(population)), 
                   size=int(len(population) * p), 
                   replace=False)
    
  i1 = np.random.choice(index, size=len(index)//2, replace=False)
  i2 = np.setdiff1d(index, i1)
  
  subset_1 = population[i1]  
  subset_2 = population[i2]

  crossed_population = []

  for i, j in zip(subset_1, subset_2):
    ni = operator(i, j)
    crossed_population.extend(ni)

  return np.asarray(crossed_population)
