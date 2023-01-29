from config import population_size

def elitist(upspring, error):
    sort_idx = error.argsort()
    return upspring[sort_idx[:population_size]]

def selection(upspring, error):
    return elitist(upspring, error)