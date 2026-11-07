import unittest
import networkx as nx
import numpy as np

from knot_theory.classifier import TrefoilMLP
from knot_theory.crossings import skeleton_graph, find_crossings


class TrefoilMLPTests(unittest.TestCase):
    def test_forward_and_predict_shapes(self):
        model = TrefoilMLP(input_size=4, seed=1)
        x = np.ones((3, 4), dtype=np.float32)
        probabilities, labels = model.predict(x)
        self.assertEqual(probabilities.shape, (3, 1))
        self.assertEqual(labels.shape, (3, 1))

    def test_training_updates_parameters(self):
        model = TrefoilMLP(input_size=4, seed=1)
        x = np.asarray([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.float32)
        y = np.asarray([[0], [1]], dtype=np.float32)
        before = model.weights[0].copy()
        model.train(x, y, epochs=2, learning_rate=0.01, log_every=0)
        self.assertFalse(np.array_equal(before, model.weights[0]))


class CrossingTests(unittest.TestCase):
    def test_cross_shape_has_one_candidate_cluster(self):
        skeleton = np.zeros((7, 7), dtype=np.uint8)
        skeleton[3, 1:6] = 1
        skeleton[1:6, 3] = 1
        graph = skeleton_graph(skeleton)
        self.assertGreaterEqual(len(find_crossings(graph, radius=2)), 1)


if __name__ == "__main__":
    unittest.main()
