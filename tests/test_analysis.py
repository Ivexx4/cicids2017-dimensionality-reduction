import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from cicids_pca.analysis import analyze_pca


class AnalyzePcaTests(unittest.TestCase):
    def test_hotelling_and_q_residual_match_kaiser_model(self):
        rng = np.random.default_rng(42)
        values = rng.normal(size=(80, 6))
        values[0, 0] = 9.0
        features = pd.DataFrame(values, columns=[f"feature_{i}" for i in range(6)])

        results = analyze_pca(features)

        scaled = StandardScaler().fit_transform(values)
        components = results.full_pca.components_[:results.kaiser_count]
        eigenvalues = results.full_pca.explained_variance_[:results.kaiser_count]
        scores = scaled @ components.T
        expected_t_squared = np.sum(scores**2 / eigenvalues, axis=1)
        expected_q_residual = np.maximum(
            np.sum(scaled**2, axis=1) - np.sum(scores**2, axis=1),
            0.0,
        )

        self.assertEqual(len(results.top_hotelling), 10)
        self.assertEqual(len(results.top_q_residual), 10)
        self.assertEqual(
            results.top_hotelling["Posição"].iloc[0],
            int(np.argmax(expected_t_squared)),
        )
        self.assertEqual(
            results.top_q_residual["Posição"].iloc[0],
            int(np.argmax(expected_q_residual)),
        )
        self.assertAlmostEqual(
            results.top_hotelling["Hotelling T²"].iloc[0],
            float(np.max(expected_t_squared)),
        )
        self.assertAlmostEqual(
            results.top_q_residual["Q-residual"].iloc[0],
            float(np.max(expected_q_residual)),
        )


if __name__ == "__main__":
    unittest.main()
