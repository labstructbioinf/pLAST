"""Missing HMMscan hits must remain JSON nulls with Pandas string columns."""

import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from plast.plast import PLAST


class EggnogMissingHitsTests(unittest.TestCase):
    def test_missing_hits_preserve_orf_order_and_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            plast = PLAST(
                data=SimpleNamespace(config={"hmmscan_db": "unused.hmm", "tmp_dir": tmp_dir}),
                logger=Mock(),
            )
            plast.parsed = pd.DataFrame(
                {
                    "type": ["CDS"] * 3,
                    "start": [30, 10, 20],
                    "locus_tag": ["orf_5", "orf_3", "orf_4"],
                    "translation": ["MKK"] * 3,
                }
            )
            hits = pd.DataFrame(
                {"query_name": ["orf_5", "orf_3"], "hit": ["33ICR", "COG1502"]}
            )
            process = Mock(returncode=0)
            process.communicate.return_value = ("", "")
            with (
                patch("plast.plast.warm_page_cache"),
                patch("plast.plast.subprocess.Popen", return_value=process),
                patch("plast.plast.read_hmmscan_output", return_value=hits),
            ):
                plast.assign_eggnog_annot(processes=1)

        self.assertEqual(plast.vector, ["COG1502", None, "33ICR"])
        self.assertEqual(json.loads(json.dumps(plast.vector, allow_nan=False)), plast.vector)
        plast.logger.info.assert_any_call(
            "HMMscan assignment completed: assigned=2, missing=1, total=3"
        )


if __name__ == "__main__":
    unittest.main()
