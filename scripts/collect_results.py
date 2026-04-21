#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect per-task kNN results")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    json_files = sorted(args.input_dir.glob("results_*.json"))
    if not json_files:
        raise ValueError(f"no per-task result files in: {args.input_dir}")

    rows: list[dict] = []
    total_records = 0
    for path in json_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        total_records += int(payload.get("num_records", 0))
        rows.extend(payload.get("results", []))

    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    macro = df["score"].dropna().mean() if ("score" in df.columns and df["score"].notna().any()) else None
    out = {
        "num_records": total_records,
        "num_tasks": int(df["task"].nunique()) if "task" in df.columns else 0,
        "macro_average": None if macro is None else float(macro),
        "results": rows,
    }
    args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
