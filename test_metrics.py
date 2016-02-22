from __future__ import division, print_function
import numpy as np
import unittest

import metrics


class PathDetectionTestCase(unittest.TestCase):

    def test_path_detection_in_example_graph(self):
        """
        Make a simple graph with known paths through it of various lengths, and
        make sure that our path detection algorithm retrieves them correctly.
        """

        # this defines the directed graph cxns (row = targ, col = source)
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

    def test_most_probable_paths_get_found_correctly(self):
        """
        Make a simple graph that has some paths that are much more probable than others,
        and make sure these can be successfully sound.
        """

        weights = np.array([
            [0, 0, 0, 2, 0],
            [2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.01],
            [0, 2, 0, 0, 0],
            [0, 0, .01, 0, 0],
        ])

        length = 3
        gain = 2
        n = 3

        correct_paths = [
            (0, 1, 3, 0),
            (1, 3, 0, 1),
            (3, 0, 1, 3),
        ]

        paths = metrics.most_probable_paths(weights, gain, length, n)

        self.assertEqual(len(correct_paths), len(paths))
        self.assertEqual(set(correct_paths), set(paths))


if __name__ == '__main__':
    unittest.main()