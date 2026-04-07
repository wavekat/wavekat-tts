#!/usr/bin/env python3
"""
Update the benchmark table in README.md from bench/results/*.csv.

CSV format (produced by `bench_rtf --csv`):
  sample,chars,iteration,synth_secs,audio_secs,rtf

Usage:
  python scripts/update_bench_table.py
  python scripts/update_bench_table.py --check   # exit 1 if README would change
"""

import csv
import re
import sys
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "bench" / "results"
README = REPO_ROOT / "README.md"

START_MARKER = "<!-- bench:start -->"
END_MARKER = "<!-- bench:end -->"

# Canonical sample order.
SAMPLES = ["short", "medium", "long"]

# Pretty labels for known config names.  Unknown names fall back to the stem.
LABEL_MAP = {
    "cpu-int4":      "CPU · int4",
    "cpu-fp32":      "CPU · fp32",
    "cuda-t4-int4":  "CUDA T4 · int4",
    "cuda-t4-fp32":  "CUDA T4 · fp32",
    "trt-t4-int4":   "TensorRT T4 · int4",
    "trt-t4-fp32":   "TensorRT T4 · fp32",
}

SORT_ORDER = list(LABEL_MAP.keys())


def label_for(stem: str) -> str:
    return LABEL_MAP.get(stem, stem.replace("-", " ").title())


def read_csv(path: Path) -> dict:
    """Return {sample: {rtf: [floats], synth_secs: [floats]}}."""
    data: dict = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            s = row["sample"]
            data.setdefault(s, {"rtf": [], "synth_secs": []})
            data[s]["rtf"].append(float(row["rtf"]))
            data[s]["synth_secs"].append(float(row["synth_secs"]))
    return data


def build_table(configs: list) -> str:
    present = [s for s in SAMPLES if any(s in d for _, d in configs)]
    if not present:
        return "_No results yet._"

    header = "| Config |" + "".join(f" RTF {s} |" for s in present)
    sep    = "|--------|" + "".join(":-----------:|" for _ in present)

    rows = []
    for stem, data in configs:
        cells = []
        for s in present:
            if s in data and data[s]["rtf"]:
                rtf = mean(data[s]["rtf"])
                # Bold values below real-time.
                cells.append(f"**{rtf:.2f}**" if rtf < 1.0 else f"{rtf:.2f}")
            else:
                cells.append("—")
        rows.append(f"| {label_for(stem)} |" + "".join(f" {c} |" for c in cells))

    lines = [header, sep] + rows + [
        "",
        "_RTF < 1.0 = faster-than-real-time. Lower is better._  ",
        "_To update: run `make bench-csv-cuda` on a T4, then commit `bench/results/`._",
    ]
    return "\n".join(lines)


def update_readme(table: str, check: bool = False) -> bool:
    """Replace the section between markers.  Returns True if the file changed."""
    text = README.read_text()
    pattern = re.compile(
        re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER),
        re.DOTALL,
    )
    replacement = f"{START_MARKER}\n{table}\n{END_MARKER}"
    new_text, n = pattern.subn(replacement, text)
    if n == 0:
        print(f"error: markers not found in {README.name}", file=sys.stderr)
        print(f"  Add  {START_MARKER!r}  and  {END_MARKER!r}  to README.md", file=sys.stderr)
        sys.exit(1)
    if new_text == text:
        return False
    if not check:
        README.write_text(new_text)
    return True


def main():
    check_mode = "--check" in sys.argv

    csvs = sorted(RESULTS_DIR.glob("*.csv"))
    if not csvs:
        print("No CSV files found in bench/results/ — nothing to do.", file=sys.stderr)
        sys.exit(0)

    configs = []
    for path in csvs:
        try:
            configs.append((path.stem, read_csv(path)))
        except Exception as exc:
            print(f"warning: skipping {path.name}: {exc}", file=sys.stderr)

    def sort_key(item):
        try:
            return (SORT_ORDER.index(item[0]), item[0])
        except ValueError:
            return (len(SORT_ORDER), item[0])

    configs.sort(key=sort_key)

    table = build_table(configs)
    changed = update_readme(table, check=check_mode)

    if check_mode:
        if changed:
            print("README.md is out of date — run `python scripts/update_bench_table.py`")
            sys.exit(1)
        else:
            print("README.md is up to date.")
    else:
        print("README.md updated." if changed else "README.md unchanged.")


if __name__ == "__main__":
    main()
