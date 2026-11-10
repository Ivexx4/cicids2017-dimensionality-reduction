"""Tests for dataset cleaning and numeric feature preparation."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from cicids_pca.data import load_and_clean_data, prepare_features


class PrepareFeaturesTests(unittest.TestCase):
    """Verify missing-value handling and removal of unusable predictors."""

    def test_handles_infinite_missing_empty_and_constant_columns(self):
        """Impute non-finite values and remove empty and constant columns."""
        frame = pd.DataFrame(
            {
                **{f"feature_{i}": [1.0, 2.0, 3.0, 4.0] for i in range(51)},
                "feature_0": [1.0, np.inf, 3.0, 4.0],
                "feature_1": [1.0, np.nan, 3.0, 4.0],
                "feature_2": [7.0, 7.0, 7.0, 7.0],
                "feature_3": [np.nan, np.nan, np.nan, np.nan],
                "Destination Port": [80, 443, 80, 22],
                "Attack Type": ["Normal", "DoS", "Normal", "DDoS"],
            }
        )

        features, target, names, summary = prepare_features(frame, "Attack Type")

        self.assertEqual(features.shape, (4, 49))
        self.assertNotIn("feature_2", names)
        self.assertNotIn("feature_3", names)
        self.assertEqual(summary.infinite_values, 1)
        self.assertEqual(summary.missing_values_imputed, 2)
        self.assertEqual(summary.all_missing_columns, ["feature_3"])
        self.assertEqual(summary.constant_columns, ["feature_2"])
        self.assertTrue(np.isfinite(features.to_numpy()).all())
        self.assertEqual(target.tolist(), ["Normal", "DoS", "Normal", "DDoS"])

    def test_raises_when_no_valid_predictors_remain(self):
        """Reject a dataset when all numeric predictors are constant."""
        frame = pd.DataFrame(
            {
                **{f"feature_{i}": [1.0, 1.0] for i in range(51)},
                "Attack Type": ["Normal", "DoS"],
            }
        )
        with self.assertRaisesRegex(ValueError, "Não restaram variáveis válidas"):
            prepare_features(frame, "Attack Type")


class LoadAndCleanDataTests(unittest.TestCase):
    """Verify chunked loading and invalid-flow filtering."""

    def test_removes_union_of_invalid_flow_rules_and_counts_classes(self):
        """Remove the union of invalid-flow cases and preserve class counts."""
        frame = pd.DataFrame(
            {
                "Init_Win_bytes_forward": [-1, 0, 0, 0],
                "Init_Win_bytes_backward": [-1, 0, -1, 0],
                "min_seg_size_forward": [0, -1, 0, 0],
                "Flow Bytes/s": [1.0, 2.0, 3.0, 4.0],
                "Attack Type": ["Normal", "DoS", "DDoS", "Normal"],
            }
        )
        with TemporaryDirectory() as temporary_directory:
            csv_path = Path(temporary_directory) / "flows.csv"
            frame.to_csv(csv_path, index=False)
            cleaned, target_col, summary = load_and_clean_data(csv_path, chunksize=2)

        self.assertEqual(target_col, "Attack Type")
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(summary.original_rows, 4)
        self.assertEqual(summary.removed_rows, 3)
        self.assertEqual(summary.forward_minus_one, 1)
        self.assertEqual(summary.backward_minus_one, 2)
        self.assertEqual(summary.negative_min_segment, 1)
        self.assertEqual(summary.class_counts_original["Normal"], 2)


if __name__ == "__main__":
    unittest.main()
