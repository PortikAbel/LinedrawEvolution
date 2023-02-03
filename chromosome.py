import numpy as np

from skimage.draw import line as discrete_line


class Chromosome:

    def __init__(self, origins, angles, lengths, target, **kwargs):
        self.origins = origins
        self.angles = angles
        self.lengths = lengths
        self.target = target
        self._fitness_scores = np.zeros(self.origins.shape[0])
        self._modified = np.ones(self.origins.shape[0]).astype(bool)
        self._fitness = None
        self._lines = self.genes_to_lines(self.origins, self.angles, self.lengths)

    def fitness(self) -> float:
        if self._modified.any():
            self._calc_fitness_scores()
            self._fitness = self._fitness_scores.sum()
        return self._fitness

    def _calc_fitness_scores(self) -> np.ndarray:
        """
        The difference of an individual by SVD
        """

        m_origins = self.origins[self._modified]
        m_angles = self.angles[self._modified]
        m_lengths = self.lengths[self._modified]

        modified_lines = self.genes_to_lines(m_origins, m_angles, m_lengths)

        scores = []

        for i, line in enumerate(modified_lines):
            scores.append(self.line_score(line, self.target))

        self._fitness_scores[self._modified] = np.asarray(scores)

        # self._modified = np.zeros(self.origins.shape[0]).astype(bool)

    @staticmethod
    def line_score(line, target):

        white = target.max()

        start_row, end_row = min(line[0][0], line[1][0]), max(line[0][0], line[1][0]) + 1
        start_col, end_col = min(line[0][1], line[1][1]), max(line[0][1], line[1][1]) + 1
        row_count = end_row - start_row
        col_count = end_col - start_col

        line_offset = line - [start_row, start_col]

        line_in_matrix = np.zeros((row_count, col_count))
        line_in_matrix[discrete_line(
            *line_offset.reshape(-1)
        )] = white

        # target_crop = np.zeros_like(line_in_matrix)
        target_crop = target[start_row:end_row, start_col:end_col]

        svd_1 = np.linalg.svd(line_in_matrix)
        svd_1 = np.concatenate((svd_1[0].reshape(-1), svd_1[2].reshape(-1)))
        svd_2 = np.linalg.svd(target_crop)
        svd_2 = np.concatenate((svd_2[0].reshape(-1), svd_2[2].reshape(-1)))

        return np.average(np.abs(np.subtract(svd_1, svd_2)))

    @staticmethod
    def genes_to_lines(origins, angles, lengths):

        assert origins.shape[0] == angles.shape[0] == lengths.shape[0]

        directions = np.stack(((np.cos(angles) * lengths // 2).astype(int),
                               (np.sin(angles) * lengths // 2).astype(int)), axis=1)

        lines = np.stack((origins - directions,
                          origins + directions), axis=1).astype(int)

        return lines
