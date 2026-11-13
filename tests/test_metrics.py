"""Small synthetic fixtures verifying metric correctness and JSON serialisation."""
import json
import unittest

import numpy as np
from evaluation.metrics import compute_all_metrics


class MetricsTests(unittest.TestCase):
    def test_json_safe_perfect_three_class(self):
        labels = np.array([0, 1, 2, 0, 1, 2])
        probs = np.eye(3)[labels]
        result = compute_all_metrics(labels, labels, probs, num_classes=3)
        self.assertEqual(result["sample_count"], 6)
        self.assertEqual(result["overall_accuracy"], 1.0)
        self.assertAlmostEqual(result["macro_roc_auc"], 1.0)
        json.dumps(result, allow_nan=False)

    def test_missing_class_has_no_full_multiclass_auc(self):
        result = compute_all_metrics([0, 0, 1, 1], [0, 1, 1, 1],
                                     [[0.9, 0.1, 0.0], [0.2, 0.8, 0.0],
                                      [0.1, 0.9, 0.0], [0.1, 0.9, 0.0]], num_classes=3)
        self.assertIsNone(result["macro_roc_auc"])
        self.assertEqual(result["per_class"]["2"]["support"], 0)
        self.assertIsNone(result["per_class"]["2"]["recall"])
        json.dumps(result, allow_nan=False)

    def test_empty_and_nan_rejected(self):
        with self.assertRaises(ValueError):
            compute_all_metrics([], [], np.zeros((0, 2)), num_classes=2)
        with self.assertRaises(ValueError):
            compute_all_metrics([0, 1], [0, 1], [[float("nan"), 0], [0, 1]], num_classes=2)

    def test_bad_probability_matrix_rejected(self):
        with self.assertRaises(ValueError):
            compute_all_metrics([0], [0], [[0.2, 0.2]], num_classes=2)


if __name__ == "__main__":
    unittest.main()
