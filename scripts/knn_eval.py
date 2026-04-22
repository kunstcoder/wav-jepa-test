#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from extract_wavjepa_features import WavJEPAInferenceWrapper, collate_with_padding


@dataclass
class Record:
    sample_id: str
    label: str
    split: str
    task: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run end-to-end WavJEPA -> kNN evaluation (in-memory)")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "torchscript", "python", "python-ckpt", "python-safetensors"],
        help="Model loading backend passed to WavJEPAInferenceWrapper",
    )
    p.add_argument("--module", type=str, default="",
                   help="Python module path. Optional when official wavjepa/sjepa package is installed.")
    p.add_argument("--class-name", type=str, default="",
                   help="Model class name. Optional when official wavjepa/sjepa package is installed.")
    p.add_argument("--data-path", type=Path, required=True, help="Dataset root containing splits.csv and audio/")
    p.add_argument("--encoder", type=str, required=True, choices=["context", "target", "auto"])
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/knn_eval"))
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--metric", type=str, default="cosine")
    p.add_argument("--weighting", type=str, default="distance", choices=["uniform", "distance"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--audio-exts", type=str, default=".wav,.flac,.mp3,.ogg,.m4a")
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


def load_split_from_dirs(data_path: Path, audio_root: Path, extensions: set[str]) -> list[Record]:
    records: list[Record] = []
    for split in ("train", "test"):
        # Support both <data>/audio/train and <data>/train layouts.
        candidates = [audio_root / split, data_path / split]
        seen: set[Path] = set()
        split_dirs: list[Path] = []
        for c in candidates:
            if c not in seen and c.exists() and c.is_dir():
                split_dirs.append(c)
                seen.add(c)

        for split_dir in split_dirs:
            for wav in sorted(split_dir.rglob("*")):
                if not wav.is_file() or wav.suffix.lower() not in extensions:
                    continue

                meta = wav.with_suffix(".json")
                label: str | None = None
                if meta.exists() and meta.is_file():
                    try:
                        obj = json.loads(meta.read_text(encoding="utf-8"))
                        raw = obj.get("labels", obj.get("label"))
                        if raw is not None:
                            label = str(raw).strip()
                    except (json.JSONDecodeError, OSError):
                        label = None

                if not label:
                    # Fallback for class-folder layouts.
                    if wav.parent != split_dir:
                        label = wav.parent.name
                    else:
                        continue

                sample_id = str(wav.relative_to(audio_root))
                records.append(Record(sample_id=sample_id, label=label, split=split, task="default"))
    return records


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    wav, _ = librosa.load(path, sr=sample_rate, mono=True)
    if wav.size == 0:
        raise ValueError(f"empty audio: {path}")
    return wav.astype(np.float32, copy=False)


def build_audio_index(audio_root: Path, extensions: set[str]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(audio_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        stem = path.stem
        if stem not in index:
            index[stem] = path
    return index


def resolve_audio_path(sample_id: str, audio_root: Path, audio_index: dict[str, Path], extensions: set[str]) -> Path | None:
    raw = audio_root / sample_id
    if raw.is_file():
        return raw

    sample_path = Path(sample_id)
    if sample_path.suffix:
        with_ext = audio_root / sample_path
        if with_ext.is_file():
            return with_ext
    else:
        for ext in extensions:
            candidate = audio_root / f"{sample_id}{ext}"
            if candidate.is_file():
                return candidate

    return audio_index.get(sample_path.stem)


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
    torch.manual_seed(args.seed)

    splits_path = args.data_path / "splits.csv"
    audio_root = args.data_path / "audio"
    if not audio_root.exists() or not audio_root.is_dir():
        audio_root = args.data_path

    if not args.model_path.exists():
        raise FileNotFoundError(f"model file not found: {args.model_path}")
    if not audio_root.exists() or not audio_root.is_dir():
        raise FileNotFoundError(f"audio/data directory not found: {audio_root}")

    extensions = {ext.strip().lower() for ext in args.audio_exts.split(",") if ext.strip()}
    if splits_path.exists():
        records = load_split(splits_path)
    else:
        records = load_split_from_dirs(args.data_path, audio_root, extensions)
        if not records:
            raise FileNotFoundError(
                f"splits.csv not found and no usable records discovered from {args.data_path}/(train|test)"
            )
    audio_index = build_audio_index(audio_root, extensions)

    wrapper = WavJEPAInferenceWrapper(
        backend=args.backend,
        model_path=str(args.model_path),
        device=args.device,
        module=args.module,
        class_name=args.class_name,
        encoder_output=args.encoder,
        hf_model_id="",
        hf_filename="model.safetensors",
    )

    valid_records: list[Record] = []
    features: list[np.ndarray] = []

    for start in tqdm(range(0, len(records), args.batch_size), desc="encode"):
        batch_records = records[start : start + args.batch_size]
        batch_audio: list[np.ndarray] = []
        batch_meta: list[Record] = []

        for rec in batch_records:
            audio_path = resolve_audio_path(rec.sample_id, audio_root, audio_index, extensions)
            if audio_path is None:
                continue
            batch_audio.append(load_audio(audio_path, args.sample_rate))
            batch_meta.append(rec)

        if not batch_audio:
            continue

        batch, lengths = collate_with_padding(batch_audio)
        batch = batch.to(wrapper.device)
        lengths = lengths.to(wrapper.device)

        emb = wrapper.encode(batch, lengths).detach().cpu().numpy().astype(np.float32, copy=False)
        for i, rec in enumerate(batch_meta):
            vec = emb[i].reshape(-1)
            if not np.isfinite(vec).all() or vec.size == 0:
                continue
            valid_records.append(rec)
            features.append(vec)

    if not features:
        raise ValueError("no valid embeddings generated; check data-path, splits.csv ids, and audio files")

    x = np.stack(features)
    tasks = sorted({r.task for r in valid_records})
    rows = [evaluate_task(x, valid_records, t, args.k, args.metric, args.weighting) for t in tasks]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "results.csv", index=False)

    macro = np.mean([r["score"] for r in rows if r["score"] is not None]) if any(r["score"] is not None for r in rows) else None
    payload = {
        "num_records": len(valid_records),
        "num_tasks": len(tasks),
        "macro_average": None if macro is None else float(macro),
        "results": rows,
    }
    (args.output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
