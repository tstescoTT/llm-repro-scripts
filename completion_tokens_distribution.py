#!/usr/bin/env python3

import argparse
import json
import re
import csv
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Union


JsonType = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


def iter_top_level_records(file_path: Path) -> Generator[JsonType, None, None]:
    """
    Yield top-level JSON records from the file. Supports:
    - A single JSON array of objects
    - JSON Lines (NDJSON), one JSON object per line
    - A single JSON object (treated as one record)
    """
    with file_path.open("r", encoding="utf-8") as f:
        # Peek first non-whitespace character to decide strategy
        # We read the whole content only if it's a JSON array/object.
        # Otherwise, we fallback to line-by-line parsing (NDJSON).
        start = f.read(4096)
        if not start:
            return
        first_non_ws = next((ch for ch in start if not ch.isspace()), "")

        # Reset to beginning for full parse if needed
        f.seek(0)

        if first_non_ws in ("[", "{"):
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # Fall back to NDJSON if full parse fails
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
                return

            if isinstance(data, list):
                for item in data:
                    yield item
            else:
                # Single JSON object treated as one record
                yield data
        else:
            # NDJSON
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def find_completion_tokens(record: JsonType) -> Iterable[int]:
    """
    Recursively search a record for any occurrences of
    response.usage.completion_tokens and yield the integer values found.
    """
    if isinstance(record, dict):
        # Direct match: record['response']['usage']['completion_tokens']
        response = record.get("response")
        if isinstance(response, dict):
            usage = response.get("usage")
            if isinstance(usage, dict):
                value = usage.get("completion_tokens")
                if isinstance(value, int):
                    yield value

        # Recurse into dict values
        for value in record.values():
            yield from find_completion_tokens(value)
    elif isinstance(record, list):
        for item in record:
            yield from find_completion_tokens(item)
    # Primitives: nothing to do


def compute_frequency_distribution(file_path: Path) -> Counter:
    counts: Counter = Counter()
    for record in iter_top_level_records(file_path):
        for tokens in find_completion_tokens(record):
            counts[tokens] += 1
    return counts


def regex_scan_completion_tokens(file_path: Path) -> Counter:
    """
    Streaming, structure-agnostic fallback that scans the file text for
    occurrences of "completion_tokens": <int> and aggregates counts.
    Useful when the file is a large, slightly malformed JSON array.
    """
    counts: Counter = Counter()
    pattern = re.compile(r"\"completion_tokens\"\s*:\s*(\d+)")
    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            for m in pattern.finditer(line):
                counts[int(m.group(1))] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute frequency distribution of response.usage.completion_tokens "
            "from a JSON/NDJSON file."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=False,
        default=Path(
            "/home/tstesco/software/llm-repro-scripts/output/"
            "concurrent_llm_requests_20250918-223217.json"
        ),
        help="Path to the JSON or NDJSON file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally limit output to the first N token values when sorted ascending.",
    )
    parser.add_argument(
        "--as-json",
        action="store_true",
        help="Output the distribution as a single JSON object (token->count).",
    )
    parser.add_argument(
        "--correlation",
        action="store_true",
        help=(
            "Compute Pearson correlation between (index // --index-div) and "
            "completion_tokens using streaming file order."
        ),
    )
    parser.add_argument(
        "--index-div",
        type=int,
        default=8,
        help="Divisor D for index // D when computing correlation (default: 8).",
    )
    parser.add_argument(
        "--batch-start",
        type=int,
        default=0,
        help=(
            "Value to add to (index // --index-div) when computing correlation. "
            "Use 1 to make the first batch equal to 1."
        ),
    )
    parser.add_argument(
        "--check-batch-uniform",
        action="store_true",
        help=(
            "Check if all completion_tokens within each batch (size --batch-size) "
            "are identical, using streaming file order."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for grouping by index // batch_size (default: 8).",
    )
    parser.add_argument(
        "--write-csv",
        type=Path,
        help=(
            "Write CSV with columns: index,completion_tokens in streaming file order."
        ),
    )
    parser.add_argument(
        "--index-base",
        type=int,
        default=0,
        help="Index base for CSV output (0 or 1). Default: 0",
    )
    args = parser.parse_args()

    counts = compute_frequency_distribution(args.input)
    if not counts:
        # Fallback to regex scan if structured parsing yielded nothing
        counts = regex_scan_completion_tokens(args.input)

    if args.correlation:
        # Streaming extraction in file order
        token_pattern = re.compile(r"\"completion_tokens\"\s*:\s*(\d+)")
        values: List[int] = []
        with args.input.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                for m in token_pattern.finditer(line):
                    values.append(int(m.group(1)))

        n = len(values)
        if n == 0:
            print("No completion_tokens found; correlation undefined")
            return

        # Build X = index // div, Y = tokens
        div = max(1, args.index_div)
        X = [i // div + args.batch_start for i in range(n)]
        Y = values

        # Compute Pearson correlation
        mean_x = sum(X) / n
        mean_y = sum(Y) / n
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(X, Y))
        var_x = sum((x - mean_x) ** 2 for x in X)
        var_y = sum((y - mean_y) ** 2 for y in Y)

        if var_x == 0 or var_y == 0:
            print("Correlation undefined (zero variance in one vector)")
            return

        r = cov / (var_x ** 0.5 * var_y ** 0.5)
        print(
            json.dumps(
                {
                    "n": n,
                    "index_div": div,
                    "pearson_r": r,
                    "mean_index_div": mean_x,
                    "mean_completion_tokens": mean_y,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.check_batch_uniform:
        token_pattern = re.compile(r"\"completion_tokens\"\s*:\s*(\d+)")
        values: List[int] = []
        with args.input.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                for m in token_pattern.finditer(line):
                    values.append(int(m.group(1)))

        n = len(values)
        if n == 0:
            print(json.dumps({"message": "No completion_tokens found"}))
            return

        bsize = max(1, args.batch_size)
        num_batches = (n + bsize - 1) // bsize
        num_full_batches = n // bsize
        num_uniform_batches = 0
        num_non_uniform_batches = 0
        non_uniform_examples: Dict[int, Dict[str, Any]] = {}

        for b in range(num_batches):
            start = b * bsize
            end = min(start + bsize, n)
            batch = values[start:end]
            distinct = sorted(set(batch))
            if len(distinct) == 1:
                num_uniform_batches += 1
            else:
                num_non_uniform_batches += 1
                if len(non_uniform_examples) < 5:
                    non_uniform_examples[b] = {
                        "start_index": start,
                        "end_index_exclusive": end,
                        "distinct_values": distinct[:10],
                    }

        result = {
            "batch_size": bsize,
            "n_values": n,
            "num_batches": num_batches,
            "num_full_batches": num_full_batches,
            "num_uniform_batches": num_uniform_batches,
            "num_non_uniform_batches": num_non_uniform_batches,
            "uniform_ratio": (
                num_uniform_batches / num_batches if num_batches else None
            ),
            "all_batches_uniform": num_non_uniform_batches == 0,
            "non_uniform_examples": non_uniform_examples,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.check_batch_uniform:
        token_pattern = re.compile(r"\"completion_tokens\"\s*:\s*(\d+)")
        values: List[int] = []
        with args.input.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                for m in token_pattern.finditer(line):
                    values.append(int(m.group(1)))

        n = len(values)
        if n == 0:
            print(json.dumps({"message": "No completion_tokens found"}))
            return

        bsize = max(1, args.batch_size)
        num_batches = (n + bsize - 1) // bsize
        num_full_batches = n // bsize
        num_uniform_batches = 0
        num_non_uniform_batches = 0
        non_uniform_examples: Dict[int, Dict[str, Any]] = {}

        for b in range(num_batches):
            start = b * bsize
            end = min(start + bsize, n)
            batch = values[start:end]
            distinct = sorted(set(batch))
            if len(distinct) == 1:
                num_uniform_batches += 1
            else:
                num_non_uniform_batches += 1
                if len(non_uniform_examples) < 5:
                    non_uniform_examples[b] = {
                        "start_index": start,
                        "end_index_exclusive": end,
                        "distinct_values": distinct[:10],
                    }

        result = {
            "batch_size": bsize,
            "n_values": n,
            "num_batches": num_batches,
            "num_full_batches": num_full_batches,
            "num_uniform_batches": num_uniform_batches,
            "num_non_uniform_batches": num_non_uniform_batches,
            "uniform_ratio": (
                num_uniform_batches / num_batches if num_batches else None
            ),
            "all_batches_uniform": num_non_uniform_batches == 0,
            "non_uniform_examples": non_uniform_examples,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.write_csv:
        # Stream and write CSV of (index, completion_tokens)
        token_pattern = re.compile(r"\"completion_tokens\"\s*:\s*(\d+)")
        index_base = 1 if args.index_base == 1 else 0
        count = 0
        args.write_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.write_csv.open("w", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["index", "completion_tokens"])
            i = 0
            with args.input.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    for m in token_pattern.finditer(line):
                        writer.writerow([i + index_base, int(m.group(1))])
                        i += 1
                        count += 1
        print(
            json.dumps(
                {
                    "csv_path": str(args.write_csv),
                    "rows": count,
                    "index_base": index_base,
                }
            )
        )
        return

    if args.as_json:
        # Serialize with string keys to ensure valid JSON
        as_obj: Dict[str, int] = {str(k): v for k, v in sorted(counts.items())}
        print(json.dumps(as_obj, indent=2, sort_keys=True))
        return

    # Human-readable text output
    print(f"File: {args.input}")
    print("completion_tokens -> count")
    printed = 0
    for token_value, count in sorted(counts.items()):
        if args.limit is not None and printed >= args.limit:
            break
        print(f"{token_value} -> {count}")
        printed += 1


if __name__ == "__main__":
    main()


