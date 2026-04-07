#!/usr/bin/env python3
"""Convert disentangle.py predictions plus IRC transcripts into transcript JSONL.

This is a post-processing utility for model output produced by `src/disentangle.py`.
It reads a prediction file in the standard disentanglement format:

    NAME.annotation.txt:QUERY_INDEX LINK_INDEX -

and combines those predicted links with the original `.ascii.txt` transcript lines
to write JSONL records shaped like:

    {"id": <int>, "message": <str>, "outgoing_edge_target_id": <int>}

The script accepts either a single `.ascii.txt` file or a directory of them. In
both cases it matches conversations by basename in the same style as the main
pipeline, so paths such as `foo.annotation.txt`, `foo.ascii.txt`, and
`foo.tok.txt` all resolve to the same conversation key.

Output behavior:
- writes one JSON object per source transcript line
- uses the zero-based line number as `id`
- strips IRC timestamps and speaker tags when possible so `message` contains the
  utterance body
- converts self-links and missing predictions to `outgoing_edge_target_id = -1`
- ignores header/comment lines and unrelated warning text in the predictions file
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List


PREDICTION_RE = re.compile(r"^(?P<name>.+):(?P<query>\d+)\s+(?P<link>-?\d+)\s+-\s*$")
MESSAGE_RE = re.compile(r"^\[\d{1,2}:\d{2}\]\s+<[^>]+>\s*(?P<body>.*)$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert disentangle.py output into transcript JSONL files.")
    parser.add_argument("--predictions", required=True, help="Path to disentangle.py .out predictions file.")
    parser.add_argument("--input", required=True, help="Path to one .ascii.txt file or a directory of them.")
    parser.add_argument("--output", help="Output JSONL path for single-file mode.")
    parser.add_argument("--output-dir", help="Output directory for directory mode.")
    return parser


def resolve_name(filename: str) -> str:
    name = filename
    for ending in [".annotation.txt", ".ascii.txt", ".raw.txt", ".tok.txt"]:
        if filename.endswith(ending):
            name = filename[: -len(ending)]
    return name


def normalise_conversation_name(path: str) -> str:
    return os.path.basename(resolve_name(path))


def parse_predictions_from_lines(lines: Iterable[str]) -> Dict[str, Dict[int, int]]:
    grouped: Dict[str, Dict[int, int]] = defaultdict(dict)
    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        match = PREDICTION_RE.match(line)
        if match is None:
            continue
        name = normalise_conversation_name(match.group("name"))
        query_id = int(match.group("query"))
        link_id = int(match.group("link"))
        grouped[name][query_id] = link_id
    return dict(grouped)


def parse_predictions(predictions_path: str) -> Dict[str, Dict[int, int]]:
    with open(predictions_path, "r", encoding="utf-8") as handle:
        return parse_predictions_from_lines(handle)


def normalise_message(line: str) -> str:
    text = line.strip()
    match = MESSAGE_RE.match(text)
    if match is not None:
        return match.group("body").strip()
    return text


def discover_ascii_files(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        if not input_path.endswith(".ascii.txt"):
            raise ValueError("Single-file mode requires a .ascii.txt input: {}".format(input_path))
        return [input_path]
    if os.path.isdir(input_path):
        ascii_files = [
            os.path.join(input_path, name)
            for name in sorted(os.listdir(input_path))
            if name.endswith(".ascii.txt") and os.path.isfile(os.path.join(input_path, name))
        ]
        return ascii_files
    raise FileNotFoundError("Input path does not exist: {}".format(input_path))


def rows_for_ascii(ascii_path: str, predictions: Dict[int, int]) -> Iterable[dict]:
    with open(ascii_path, "r", encoding="utf-8") as handle:
        for idx, raw_line in enumerate(handle):
            target_id = predictions.get(idx, -1)
            if target_id == idx:
                target_id = -1
            yield {
                "id": idx,
                "message": normalise_message(raw_line),
                "outgoing_edge_target_id": target_id,
            }


def write_jsonl(output_path: str, rows: Iterable[dict]) -> None:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def output_name_for_ascii(ascii_path: str) -> str:
    return normalise_conversation_name(ascii_path) + "-transcript.jsonl"


def validate_args(args) -> None:
    is_file = os.path.isfile(args.input)
    is_dir = os.path.isdir(args.input)
    if not is_file and not is_dir:
        raise FileNotFoundError("Input path does not exist: {}".format(args.input))
    if is_file:
        if args.output is None:
            raise ValueError("Single-file mode requires --output.")
        if args.output_dir is not None:
            raise ValueError("Single-file mode does not accept --output-dir.")
    else:
        if args.output_dir is None:
            raise ValueError("Directory mode requires --output-dir.")
        if args.output is not None:
            raise ValueError("Directory mode does not accept --output.")


def convert_single_file(ascii_path: str, output_path: str, all_predictions: Dict[str, Dict[int, int]]) -> None:
    conversation_name = normalise_conversation_name(ascii_path)
    if conversation_name not in all_predictions:
        raise ValueError("No matching predictions found for {}".format(ascii_path))
    write_jsonl(output_path, rows_for_ascii(ascii_path, all_predictions[conversation_name]))


def convert_directory(input_dir: str, output_dir: str, all_predictions: Dict[str, Dict[int, int]]) -> List[str]:
    written = []
    for ascii_path in discover_ascii_files(input_dir):
        conversation_name = normalise_conversation_name(ascii_path)
        if conversation_name not in all_predictions:
            continue
        output_path = os.path.join(output_dir, output_name_for_ascii(ascii_path))
        write_jsonl(output_path, rows_for_ascii(ascii_path, all_predictions[conversation_name]))
        written.append(output_path)
    if not written:
        raise ValueError("No matching predictions found for any .ascii.txt file in {}".format(input_dir))
    return written


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    all_predictions = parse_predictions(args.predictions)

    if os.path.isfile(args.input):
        convert_single_file(args.input, args.output, all_predictions)
    else:
        convert_directory(args.input, args.output_dir, all_predictions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
