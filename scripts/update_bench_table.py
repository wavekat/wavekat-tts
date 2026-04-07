#!/usr/bin/env python3
"""
Update the benchmark table in README.md from bench/results/*.csv.

CSV format (produced by `bench_rtf --csv`):
  backend,precision,provider,hardware,date,sample,chars,iteration,synth_secs,audio_secs,rtf

Legacy format (no metadata columns) is also accepted:
  sample,chars,iteration,synth_secs,audio_secs,rtf

Usage:
  python scripts/update_bench_table.py
  python scripts/update_bench_table.py --check   # exit 1 if README would change
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "bench" / "results"
README = REPO_ROOT / "README.md"

START_MARKER = "<!-- bench:start -->"
END_MARKER = "<!-- bench:end -->"

SAMPLES = ["short", "medium", "long"]

# Display order for known (provider, hardware) combinations.
SORT_KEY = {
    ("cpu",      "unknown"): (0, ""),
    ("cuda",     "t4"):       (1, ""),
    ("cuda",     "a10g"):     (2, ""),
    ("tensorrt", "t4"):       (3, ""),
    ("tensorrt", "a10g"):     (4, ""),
    ("coreml",   "unknown"):  (5, ""),
}


def sort_key(config: dict) -> tuple:
    k = (config["provider"], config["hardware"])
    order, _ = SORT_KEY.get(k, (99, ""))
    return (config["backend"], order, config["precision"], config["hardware"])


def read_csv(path: Path) -> list[dict]:
    """
    Return a list of row dicts.  Adds default metadata for legacy files
    that don't have the new columns.
    """
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "backend" not in row:
                # Legacy format — derive metadata from filename
                stem = path.stem  # e.g. "cuda-t4-int4"
                parts = stem.split("-")
                row["backend"] = "qwen3-tts"
                row["precision"] = parts[-1] if parts[-1] in ("int4", "fp32") else "int4"
                row["provider"] = parts[0] if parts else "cpu"
                row["hardware"] = parts[1] if len(parts) > 2 else "unknown"
                row["date"] = ""
            rows.append(row)
    return rows


def group_rows(all_rows: list[dict]) -> list[tuple[dict, dict]]:
    """
    Group rows by (backend, precision, provider, hardware).
    Returns list of (config_dict, {sample: {rtf: [], synth_secs: []}}).
    """
    groups: dict[tuple, dict] = defaultdict(lambda: defaultdict(lambda: {"rtf": [], "synth_secs": []}))
    configs: dict[tuple, dict] = {}

    for row in all_rows:
        key = (row["backend"], row["precision"], row["provider"], row["hardware"])
        groups[key][row["sample"]]["rtf"].append(float(row["rtf"]))
        groups[key][row["sample"]]["synth_secs"].append(float(row["synth_secs"]))
        if key not in configs:
            configs[key] = {
                "backend":   row["backend"],
                "precision": row["precision"],
                "provider":  row["provider"],
                "hardware":  row["hardware"],
                "date":      row.get("date", ""),
            }
        elif row.get("date"):
            # Keep the most recent date seen for this config.
            if row["date"] > configs[key]["date"]:
                configs[key]["date"] = row["date"]

    result = [(configs[k], dict(groups[k])) for k in groups]
    result.sort(key=lambda x: sort_key(x[0]))
    return result


def hardware_label(hw: str) -> str:
    return {"t4": "T4", "a10g": "A10G", "unknown": "—"}.get(hw, hw)


def provider_label(pv: str) -> str:
    return {"cpu": "CPU", "cuda": "CUDA", "tensorrt": "TensorRT", "coreml": "CoreML"}.get(pv, pv)


def build_table(groups: list[tuple[dict, dict]]) -> str:
    present = [s for s in SAMPLES if any(s in data for _, data in groups)]
    if not present:
        return "_No results yet._"

    rtf_headers = "".join(f" RTF {s} |" for s in present)
    header = f"| Backend | Precision | Provider | Hardware |{rtf_headers} Date |"
    sep    = f"|---------|-----------|----------|----------|" + ":-----------:|" * len(present) + "------|"

    rows = []
    for config, data in groups:
        cells = []
        for s in present:
            if s in data and data[s]["rtf"]:
                rtf = mean(data[s]["rtf"])
                cells.append(f"**{rtf:.2f}**" if rtf < 1.0 else f"{rtf:.2f}")
            else:
                cells.append("—")

        hw   = hardware_label(config["hardware"])
        prov = provider_label(config["provider"])
        date = config["date"] or "—"
        rows.append(
            f"| {config['backend']} | {config['precision']} | {prov} | {hw} |"
            + "".join(f" {c} |" for c in cells)
            + f" {date} |"
        )

    lines = [header, sep] + rows + [
        "",
        "_RTF < 1.0 = faster-than-real-time. Lower is better._  ",
        "_To update: run `make bench-csv-cuda` on target hardware, then commit `bench/results/`._",
    ]
    return "\n".join(lines)


def update_readme(table: str, check: bool = False) -> bool:
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

    all_rows = []
    for path in csvs:
        try:
            all_rows.extend(read_csv(path))
        except Exception as exc:
            print(f"warning: skipping {path.name}: {exc}", file=sys.stderr)

    groups = group_rows(all_rows)
    table = build_table(groups)
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
