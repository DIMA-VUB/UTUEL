"""Unit tests for coordinate-preserving table permutations."""

import unittest

from generate import PermutationKind, _transform_records


class PermutationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {
                "id": "a", "table_id": "table-1", "header": ["name", "year", "value"],
                "rows": [["one", "2023", "10"], ["two", "2024", "20"]],
                "target_column": 2, "target_columns": [1, 2], "condition_columns": [0, 1], "target_rows": [1],
                "metadata": {"source": "fixture", "version": 1},
            },
            {
                "id": "b", "table_id": "table-1", "header": ["name", "year", "value"],
                "rows": [["one", "2023", "10"], ["two", "2024", "20"]],
                "target_column": 1, "condition_columns": [0], "target_rows": [0],
            },
        ]

    def test_joint_permutation_preserves_answers_and_shared_table_layout(self) -> None:
        output = _transform_records(self.records, PermutationKind.ROWS_AND_COLUMNS, seed=9)
        self.assertEqual(output[0]["header"], output[1]["header"])
        self.assertEqual(output[0]["rows"], output[1]["rows"])
        answer_row = output[0]["target_rows"][0]
        answer_column = output[0]["target_column"]
        self.assertEqual(output[0]["rows"][answer_row][answer_column], "20")
        self.assertEqual(output[0]["header"][answer_column], "value")
        self.assertEqual(
            {output[0]["header"][index] for index in output[0]["target_columns"]},
            {"year", "value"},
        )
        self.assertEqual(output[0]["permutation_seed"], 9)
        self.assertEqual(output[0]["permutation_kind"], "rows_columns")
        self.assertEqual(output[0]["metadata"], self.records[0]["metadata"])


if __name__ == "__main__":
    unittest.main()