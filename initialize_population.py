import numpy as np

from operator import itemgetter

from util import gen_random_angles

from chromosome import Chromosome


class MList(list):
    """
    Custom list that supports querying a list with a numpy array of indices
    """

    def __getitem__(self, indices):
        if isinstance(indices, np.ndarray):
            if indices.shape[0] == 1:
                return MList([itemgetter(*indices)(self)])
            return MList(itemgetter(*indices)(self))
        else:
            return super(MList, self).__getitem__(indices)

# def create_individual_1(num_lines, img_shape=(512, 512), max_line_length=10):
#     """
#     :param int num_lines: The number of lines that constitute the image
#     :param tuple img_shape: The shape of the image
#     :param int max_line_length: The maximum allowed length for lines
#     """
#     start_points = np.random.uniform(size=(num_lines, 2))
#
#     print(start_points)
#
#     start_points[:, 0] *= img_shape[0]
#     start_points[:, 1] *= img_shape[1]
#
#     num_circle_points = 5000
#     circle_radius = max_line_length
#
#     t = np.linspace(0, 2 * np.pi,
#                     num_circle_points,
#                     endpoint=False)[np.random.choice(range(num_circle_points),
#                                                      num_lines,
#                                                      replace=False)]
#
#     end_points = np.array((circle_radius * np.cos(t),
#                            circle_radius * np.sin(t))).T
#
#     end_points += start_points
#
#     return np.stack((start_points, end_points), axis=1).astype(int)
#
#
# def create_individual_2(num_lines, start_points, max_line_length=10):
#     """
#     :param int num_lines: The number of lines that constitute the image
#     :param tuple img_shape: The shape of the image
#     :param int max_line_length: The maximum allowed length for lines
#     """
#
#     start_points = start_points[np.random.choice(
#         range(len(start_points)), num_lines, replace=False)]
#
#     num_circle_points = 5000
#     circle_radius = max_line_length
#
#     t = np.linspace(0, 2 * np.pi,
#                     num_circle_points,
#                     endpoint=False)[np.random.choice(range(num_circle_points),
#                                                      num_lines,
#                                                      replace=False)]
#
#     end_points = np.array((circle_radius * np.cos(t),
#                            circle_radius * np.sin(t))).T
#
#     end_points += start_points
#
#     return np.stack((start_points, end_points), axis=1).astype(int)


def create_individual_3(num_lines, target_sparse):
    """
    Creates an individual by selecting the endpoints of lines from the active pixels of the target image.

    :param int num_lines: The number of lines that constitute the image
    :param target_sparse: The image we want to approximate
    """

    num_white_points = target_sparse.shape[0]
    indices = np.random.choice(num_white_points, (2, num_lines), replace=False)

    start_points = target_sparse[indices[0]]
    end_points = target_sparse[indices[1]]

    return np.stack((start_points, end_points), axis=1)


def init_population(population_size, create_individual=create_individual_3, **kwargs):

    return np.asarray([create_individual(**kwargs) for _ in range(population_size)])


def initialize_population(pop_size, num_lines, line_origins, in_line_length, target):
    """
    :param int pop_size: The size of the population (number of chromosomes)
    :param int num_lines: The number of lines that constitute an image (chromosome)
    :param tuple line_origins: The set of points to draw line origins from
    :param int in_line_length: The initial length for lines
    :param numpy.ndarray target: The target image
    """

    population = MList([])

    origins_sample = line_origins[np.random.choice(range(len(line_origins)), num_lines, replace=False)]

    for _ in range(pop_size):
        # origins_sample = line_origins[np.random.choice(range(len(line_origins)), num_lines, replace=False)]

        angles_sample = gen_random_angles(num_lines)

        line_length_sample = np.ones(num_lines) * in_line_length

        population.append(Chromosome(origins_sample, angles_sample, line_length_sample, target))

    return population
