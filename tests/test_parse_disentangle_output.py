#!/usr/bin/env python3

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))

import parse_disentangle_output as parser


class ParseDisentangleOutputTest(unittest.TestCase):
    def write_file(self, path, text):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)

    def read_jsonl(self, path):
        with open(path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def test_single_file_conversion_ignores_noise_and_fills_missing_predictions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ascii_path = os.path.join(tmpdir, "conv.ascii.txt")
            predictions_path = os.path.join(tmpdir, "predictions.out")
            output_path = os.path.join(tmpdir, "conv-transcript.jsonl")

            self.write_file(
                ascii_path,
                "[12:18] <nick> hello there\n=== nick has joined #ubuntu []\n[12:19] <other> hi back\n",
            )
            self.write_file(
                predictions_path,
                "# header\n"
                "The dy.parameter(...) call is now DEPRECATED.\n"
                "/tmp/conv.annotation.txt:0 0 -\n"
                "/tmp/conv.annotation.txt:2 1 -\n",
            )

            parser.main(
                [
                    "--predictions",
                    predictions_path,
                    "--input",
                    ascii_path,
                    "--output",
                    output_path,
                ]
            )

            self.assertEqual(
                self.read_jsonl(output_path),
                [
                    {"id": 0, "message": "hello there", "outgoing_edge_target_id": -1},
                    {"id": 1, "message": "=== nick has joined #ubuntu []", "outgoing_edge_target_id": -1},
                    {"id": 2, "message": "hi back", "outgoing_edge_target_id": 1},
                ],
            )

    def test_directory_conversion_writes_one_file_per_matched_conversation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "input")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(input_dir, exist_ok=True)

            self.write_file(os.path.join(input_dir, "alpha.ascii.txt"), "[12:18] <a> first\n")
            self.write_file(os.path.join(input_dir, "beta.ascii.txt"), "[12:18] <b> second\n")
            self.write_file(os.path.join(input_dir, "gamma.ascii.txt"), "[12:18] <c> third\n")

            predictions_path = os.path.join(tmpdir, "predictions.out")
            self.write_file(
                predictions_path,
                "/tmp/alpha.annotation.txt:0 0 -\n"
                "/tmp/beta.annotation.txt:0 0 -\n"
                "/tmp/extra.annotation.txt:0 0 -\n",
            )

            parser.main(
                [
                    "--predictions",
                    predictions_path,
                    "--input",
                    input_dir,
                    "--output-dir",
                    output_dir,
                ]
            )

            self.assertTrue(os.path.exists(os.path.join(output_dir, "alpha-transcript.jsonl")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "beta-transcript.jsonl")))
            self.assertFalse(os.path.exists(os.path.join(output_dir, "gamma-transcript.jsonl")))

    def test_prediction_name_normalisation_matches_annotation_and_ascii_suffixes(self):
        parsed = parser.parse_predictions_from_lines(
            [
                "/tmp/path/demo.annotation.txt:3 2 -",
                "/tmp/path/demo.ascii.txt:4 3 -",
                "/tmp/path/demo.tok.txt:5 4 -",
            ]
        )
        self.assertEqual(parsed["demo"], {3: 2, 4: 3, 5: 4})

    def test_single_file_mode_requires_matching_predictions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ascii_path = os.path.join(tmpdir, "conv.ascii.txt")
            predictions_path = os.path.join(tmpdir, "predictions.out")
            output_path = os.path.join(tmpdir, "conv-transcript.jsonl")

            self.write_file(ascii_path, "[12:18] <nick> hello there\n")
            self.write_file(predictions_path, "/tmp/other.annotation.txt:0 0 -\n")

            with self.assertRaisesRegex(ValueError, "No matching predictions found"):
                parser.main(
                    [
                        "--predictions",
                        predictions_path,
                        "--input",
                        ascii_path,
                        "--output",
                        output_path,
                    ]
                )


if __name__ == "__main__":
    unittest.main()
