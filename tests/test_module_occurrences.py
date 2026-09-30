import unittest

import numpy as np

from plast.plast import PLAST


class ModuleOccurrenceTests(unittest.TestCase):
    def setUp(self):
        self.a = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        self.b = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        self.x = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        self.query_rows = np.vstack([self.a, self.b])
        self.query_embedding = np.asarray([1.0, 1.0, 0.0], dtype=np.float32)

    def test_preserves_forward_and_reverse_non_overlapping_occurrences(self):
        target_rows = np.vstack(
            [self.a, self.b, self.x, self.a, self.b, self.x, self.b, self.a]
        )
        scan = PLAST._sliding_window_scores(
            self.query_embedding,
            target_rows,
            np.ones(len(target_rows), dtype=np.float32),
            window_size=2,
            circular=False,
        )

        hits = PLAST._module_window_hits(
            scan,
            self.query_rows,
            target_rows,
            query_tokens=["a", "b"],
            target_tokens=["a", "b", "x", "a", "b", "x", "b", "a"],
            score_delta=0.0,
        )

        self.assertEqual([hit["window_start"] for hit in hits], [0, 3, 6])
        self.assertEqual([hit["orientation"] for hit in hits], [1, 1, -1])
        self.assertEqual(
            [(pair["query_orf"], pair["target_orf"]) for pair in hits[2]["pairs"]],
            [(0, 7), (1, 6)],
        )

    def test_preserves_occurrence_wrapping_over_circular_boundary(self):
        target_rows = np.vstack([self.b, self.x, self.a])
        scan = PLAST._sliding_window_scores(
            self.query_embedding,
            target_rows,
            np.ones(len(target_rows), dtype=np.float32),
            window_size=2,
            circular=True,
        )

        hits = PLAST._module_window_hits(
            scan,
            self.query_rows,
            target_rows,
            query_tokens=["a", "b"],
            target_tokens=["b", "x", "a"],
            score_delta=0.0,
        )

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["window_start"], 2)
        self.assertEqual(hits[0]["window_end"], 0)
        self.assertTrue(hits[0]["window_wraps"])
        self.assertEqual(
            [(pair["query_orf"], pair["target_orf"]) for pair in hits[0]["pairs"]],
            [(0, 2), (1, 0)],
        )

    def test_emits_only_exact_cluster_pairs_inside_an_occurrence(self):
        target_rows = np.vstack([self.a, self.b])
        scan = PLAST._sliding_window_scores(
            self.query_embedding,
            target_rows,
            np.ones(len(target_rows), dtype=np.float32),
            window_size=2,
            circular=False,
        )

        hits = PLAST._module_window_hits(
            scan,
            self.query_rows,
            target_rows,
            query_tokens=["a", "b"],
            target_tokens=["a", "different-b"],
            score_delta=0.0,
        )

        self.assertEqual(hits[0]["query_orfs"], [0, 1])
        self.assertEqual(hits[0]["target_orfs"], [0, 1])
        self.assertEqual(
            [(pair["query_orf"], pair["target_orf"]) for pair in hits[0]["pairs"]],
            [(0, 0)],
        )

    def test_linear_scan_does_not_join_end_and_start(self):
        target_rows = np.vstack([self.b, self.x, self.a])
        scan = PLAST._sliding_window_scores(
            self.query_embedding,
            target_rows,
            np.ones(len(target_rows), dtype=np.float32),
            window_size=2,
            circular=False,
        )

        hits = PLAST._module_window_hits(
            scan,
            self.query_rows,
            target_rows,
            query_tokens=["a", "b"],
            target_tokens=["b", "x", "a"],
            score_delta=0.0,
        )

        self.assertEqual(len(scan["scores"]), 2)
        self.assertFalse(hits[0]["window_wraps"])
        self.assertNotEqual(set(hits[0]["target_orfs"]), {0, 2})


if __name__ == "__main__":
    unittest.main()
