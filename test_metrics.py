from __future__ import division, print_function
import numpy as np
import unittest

import metrics


class PathDetectionTestCase(unittest.TestCase):

    def test_path_detection_in_example_graph(self):

        weights = np.array([
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
        ])

        length_1_paths = [
            (0, 4),
            (1, 0),
            (1, 3),
            (2, 1),
            (2, 3),
            (4, 1),
        ]

        length_2_paths = [
            (0, 4, 1),
            (1, 0, 4),
            (2, 1, 0),
            (2, 1, 3),
            (4, 1, 0),
            (4, 1, 3),
        ]

        length_3_paths = [
            (0, 4, 1, 0),
            (0, 4, 1, 3),
            (1, 0, 4, 1),
            (2, 1, 0, 4),
            (4, 1, 0, 4),
        ]

        for length, true_paths in zip(range(1, 4), [length_1_paths, length_2_paths, length_3_paths]):
            paths = metrics.paths_of_length(weights, length)

            self.assertEqual(len(paths), len(true_paths))
            self.assertEqual(set(paths), set(true_paths))




if __name__ == '__main__':
    unittest.main()