import os

population_size = 50
num_epochs      = 500
line_count      = 20

image_shape = (480, 480)

image_name = "triangle.png" #"device4-10.png"
# image_name = "face.jpg" #"device4-10.png"

# input_dir = os.path.join("input", "original")
input_dir = "input"
output_dir = "output"

crossover_type = 'elitist_uniform'
selection_type = 'elite'
mutation_type = 'non_uniform'

num_lines_to_mutate = 1
num_lines_to_crossover = 5

mutation_probability = .8
crossover_probability = .2

line_ratio = .15




