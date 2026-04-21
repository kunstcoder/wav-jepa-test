#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


@dataclass
class Record:
    sample_id: str
    label: str
    split: str
    task: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run kNN evaluation from pre-extracted embeddings")
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True, help="CSV with columns: id,label,split[,task]")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--task", type=str, default="", help="Evaluate only one task name")
    p.add_argument("--results-csv", type=str, default="results.csv")
    p.add_argument("--results-json", type=str, default="results.json")
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--metric", type=str, default="cosine")
    p.add_argument("--weighting", type=str, default="distance", choices=["uniform", "distance"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_split(path: Path) -> list[Record]:
    df = pd.read_csv(path)
    required = {"id", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"split file missing columns: {sorted(missing)}")

    if "task" not in df.columns:
        df["task"] = "default"

    return [
        Record(sample_id=str(r.id), label=str(r.label), split=str(r.split), task=str(r.task))
        for r in df.itertuples(index=False)
    ]


def flatten_embedding(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        raise ValueError("empty embedding")
    if not np.isfinite(arr).all():
        raise ValueError("embedding contains NaN/Inf")
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    return arr.mean(axis=0).reshape(-1)


def load_features(features_dir: Path, records: Iterable[Record]) -> tuple[np.ndarray, list[Record]]:
    vectors: list[np.ndarray] = []
    valid_records: list[Record] = []

    for rec in records:
        fp = features_dir / f"{rec.sample_id}.npy"
        if not fp.exists():
            continue
        arr = np.load(fp)
        vec = flatten_embedding(arr)
        vectors.append(vec.astype(np.float32, copy=False))
        valid_records.append(rec)

    if not vectors:
        raise ValueError("no valid feature vectors found")

    dim = vectors[0].shape[0]
    for i, v in enumerate(vectors):
        if v.shape[0] != dim:
            raise ValueError(f"dimension mismatch at index {i}: {v.shape[0]} vs {dim}")

    return np.stack(vectors), valid_records


def evaluate_task(x: np.ndarray, recs: list[Record], task: str, k: int, metric: str, weighting: str) -> dict:
    idx = [i for i, r in enumerate(recs) if r.task == task]
    x_t = x[idx]
    r_t = [recs[i] for i in idx]

    train_idx = [i for i, r in enumerate(r_t) if r.split == "train"]
    test_idx = [i for i, r in enumerate(r_t) if r.split == "test"]

    if not train_idx or not test_idx:
        return {"task": task, "split": "test", "metric_name": "accuracy", "score": None, "reason": "missing train/test"}

    y_train = [r_t[i].label for i in train_idx]
    y_test = [r_t[i].label for i in test_idx]

    k_eff = max(1, min(k, len(train_idx)))
    clf = KNeighborsClassifier(n_neighbors=k_eff, metric=metric, weights=weighting)
    clf.fit(x_t[train_idx], y_train)
    pred = clf.predict(x_t[test_idx])
    score = float(accuracy_score(y_test, pred))
    return {"task": task, "split": "test", "metric_name": "accuracy", "score": score, "reason": ""}


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_split(args.splits)
    if args.task:
        records = [r for r in records if r.task == args.task]
        if not records:
            raise ValueError(f"no records found for task: {args.task}")

    x, valid_records = load_features(args.features_dir, records)
    tasks = sorted({r.task for r in valid_records})

    rows = [evaluate_task(x, valid_records, t, args.k, args.metric, args.weighting) for t in tasks]
    df = pd.DataFrame(rows)
    csv_path = args.output_dir / args.results_csv
    json_path = args.output_dir / args.results_json

    df.to_csv(csv_path, index=False)

    macro = df["score"].dropna().mean() if (df["score"].notna().any()) else None
    payload = {
        "num_records": len(valid_records),
        "num_tasks": len(tasks),
        "macro_average": None if macro is None else float(macro),
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
