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

        paths, probs = metrics.most_probable_paths(weights, gain, length, n)

        self.assertEqual(len(correct_paths), len(paths))
        self.assertEqual(set(correct_paths), set(paths))

        np.testing.assert_array_equal(probs, np.array(sorted(probs))[::-1])

    def test_reordering_of_columns_by_paths(self):
        """
        Make sure we can reorder our data to help visualize it.
        """

        rates_initial = np.array([
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
        ])

        paths = [
            (4, 0, 2),
            (0, 3),
            (0, 2),
        ]

        rates_correct = rates_initial[:, np.array([4, 0, 2, 3, 1])]
        reordering_correct = np.array([4, 0, 2, 3, 1])

        rates, reordering = metrics.reorder_by_paths(rates_initial, paths)

        np.testing.assert_array_equal(rates, rates_correct)
        np.testing.assert_array_equal(reordering, reordering_correct)

    def test_find_start_nodes_with_nonoverlapping_path_trees(self):
        """
        Given a network, make sure we can find a pair of starting nodes with non-overlapping path
        trees emanating from them.
        """
        weights = np.array([
            [0, 1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 1, 1, 1, 0],
        ])

        length = 2
        nodes = [0, 1, 2, 3, 4]

        node_0_correct = 0
        node_1_correct = 3

        node_0, node_1 = metrics.first_node_pair_non_overlapping_path_tree(nodes, weights, length)

        self.assertEqual(node_0, node_0_correct)
        self.assertEqual(node_1, node_1_correct)

    def test_path_tree_overlap_works_correctly(self):
        """
        Given a network, make sure we can calculate the pairwise overlap in path trees.
        """
        weights = np.array([
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ])

        nodes = range(6)
        length = 2

        path_trees_correct = [
            [(0, 1), (0, 1, 2)],
            [(1, 2), (1, 2, 0)],
            [(2, 0), (2, 0, 1)],
            [(3, 4), (3, 4, 5)],
            [(4, 5), (4, 5, 3)],
            [(5, 3), (5, 3, 4)],
        ]

        overlap_correct = np.array([
            [3, 3, 3, 0, 0, 0],
            [3, 3, 3, 0, 0, 0],
            [3, 3, 3, 0, 0, 0],
            [0, 0, 0, 3, 3, 3],
            [0, 0, 0, 3, 3, 3],
            [0, 0, 0, 3, 3, 3],
        ])

        overlap, path_trees = metrics.path_tree_overlaps(nodes, weights, length)

        np.testing.assert_array_equal(overlap, overlap_correct)
        self.assertEqual(path_trees, path_trees_correct)


class SequentialActivityPatternsTestCase(unittest.TestCase):

    def test_get_number_of_past_occurrences_of_most_recent_sequence_works_correctly(self):

        seq = np.array([0, 1, 2, 3, 5, 1, 2, 6, 0, 3, 2, 1, 2, 3, 5, 0, 2, 1, 2, 3])

        self.assertEqual(metrics.get_number_of_past_occurrences_of_most_recent_sequence(seq, 3), 2)

    def test_get_number_of_past_occurrences_works_correctly(self):

        seq = np.array([0, 1, 2, 3, 5, 1, 2, 6, 0, 3, 2, 1, 2, 3, 5, 0, 2, 1, 2, 3])
        occs_correct = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 2])

        np.testing.assert_array_equal(
            metrics.get_number_of_past_occurrences(seq, 3), occs_correct
        )

    def test_get_number_of_past_occurrences_of_specific_subsequence_works_correctly(self):

        seq = np.array([0, 1, 2, 3, 5, 1, 2, 6, 0, 3, 0, 1, 2, 3, 5, 0, 0, 1, 2, 7])
        subseq = np.array([0, 1, 2])
        occs_correct = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3])

        np.testing.assert_array_equal(
            metrics.get_number_of_past_occurrences_of_specific_sequence(seq, subseq), occs_correct,
        )


if __name__ == '__main__':
    unittest.main()