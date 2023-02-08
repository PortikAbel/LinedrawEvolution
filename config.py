import os

population_size = 10
num_epochs      = 20
line_count      = 50

image_shape = (480, 480)

image_name = "device3-1.jpg" #"device4-10.png"

# input_dir = os.path.join("input", "original")
input_dir = "input"
output_dir = "output"

crossover_type = 'uniform'
selection_type = 'elite'
mutation_type = 'non_uniform'




