import numpy as np


def create_individual_1(num_lines, img_shape=(512, 512), max_line_length=10):
    """
    :param int num_lines: The number of lines that constitute the image
    :param tuple img_shape: The shape of the image
    :param int max_line_length: The maximum allowed length for lines
    """
    start_points = np.random.uniform(size=(num_lines, 2))

    print(start_points)

    start_points[:, 0] *= img_shape[0]
    start_points[:, 1] *= img_shape[1]

    num_circle_points = 5000
    circle_radius = max_line_length

    t = np.linspace(0, 2 * np.pi,
                    num_circle_points,
                    endpoint=False)[np.random.choice(range(num_circle_points),
                                                     num_lines,
                                                     replace=False)]

    end_points = np.array((circle_radius * np.cos(t),
                           circle_radius * np.sin(t))).T

    end_points += start_points

    return np.stack((start_points, end_points), axis=1).astype(int)


def create_individual_2(num_lines, start_points, max_line_length=10):
    """
    :param int num_lines: The number of lines that constitute the image
    :param tuple img_shape: The shape of the image
    :param int max_line_length: The maximum allowed length for lines
    """

    start_points = start_points[np.random.choice(
        range(len(start_points)), num_lines, replace=False)]

    num_circle_points = 5000
    circle_radius = max_line_length

    t = np.linspace(0, 2 * np.pi,
                    num_circle_points,
                    endpoint=False)[np.random.choice(range(num_circle_points),
                                                     num_lines,
                                                     replace=False)]

    end_points = np.array((circle_radius * np.cos(t),
                           circle_radius * np.sin(t))).T

    end_points += start_points

    return np.stack((start_points, end_points), axis=1).astype(int)


def create_individual_3(num_lines, target_sparse):
    """
    Creates an individual by selecting the endpoints of lines from the active pixels of the target image.

    :param int num_lines: The number of lines that constitute the image
    :param target_img: The image we want to approximate
    """

    num_white_points = target_sparse.shape[0]
    indices = np.random.choice(num_white_points, (2, num_lines), replace=False)

    start_points = target_sparse[indices[0]]
    end_points = target_sparse[indices[1]]

    return np.stack((start_points, end_points), axis=1)


def init_population(population_size, create_individual=create_individual_3, **kwargs):

    return np.asarray([create_individual(**kwargs) for _ in range(population_size)])
