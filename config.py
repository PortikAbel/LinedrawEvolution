import os

population_size = 50
num_epochs      = 20
line_count      = 200

image_shape = (480, 480)

image_name = "face.jpg" #"device4-10.png"

input_dir = os.path.join("input", "original")
output_dir = "output"

crossover_type = 'uniform'
selection_type = 'elite'
mutation_type = 'non_uniform'




