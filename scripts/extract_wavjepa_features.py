#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from tqdm import tqdm

try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for WavJEPA adapter. Install torch first.") from exc


@dataclass
class Sample:
    sample_id: str
    audio_path: Path
    task: str


class WavJEPAInferenceWrapper:
    def __init__(
        self,
        backend: str,
        model_path: str,
        device: str,
        module: str,
        class_name: str,
    ) -> None:
        self.backend = backend
        self.device = torch.device(device)

        if backend == "torchscript":
            if not model_path:
                raise ValueError("--model-path is required when --backend=torchscript")
            self.model = torch.jit.load(model_path, map_location=self.device)
        elif backend == "python":
            if not module or not class_name:
                raise ValueError("--module and --class-name are required when --backend=python")
            mod = importlib.import_module(module)
            cls = getattr(mod, class_name)
            self.model = cls(model_path)
        elif backend == "mock":
            self.model = None
        else:
            raise ValueError(f"unsupported backend: {backend}")

        if self.model is not None:
            self.model.eval()
            if hasattr(self.model, "to"):
                self.model.to(self.device)

    @torch.inference_mode()
    def encode(self, batch_audio: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if self.backend == "mock":
            x = batch_audio
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True)
            mx = x.max(dim=1, keepdim=True).values
            mn = x.min(dim=1, keepdim=True).values
            return torch.cat([mean, std, mx, mn], dim=1)

        if self.backend == "python" and hasattr(self.model, "encode"):
            out = self.model.encode(batch_audio, lengths)
        else:
            out = self.model(batch_audio, lengths)

        if isinstance(out, dict):
            for key in ("embeddings", "embedding", "x"):
                if key in out:
                    out = out[key]
                    break
        if isinstance(out, (tuple, list)):
            out = out[0]
        if not isinstance(out, torch.Tensor):
            raise TypeError("model output must be tensor/tuple/list/dict containing tensor")

        if out.ndim == 3:
            out = out.mean(dim=1)
        elif out.ndim == 1:
            out = out.unsqueeze(0)

        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract WavJEPA embeddings and export X-ARES-compatible features")
    p.add_argument("--manifest", type=Path, required=True, help="CSV with columns: id,audio_path[,task]")
    p.add_argument("--output-dir", type=Path, required=True, help="Output feature directory")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0, help="reserved for future use")
    p.add_argument("--backend", type=str, default="torchscript", choices=["torchscript", "python", "mock"])
    p.add_argument("--model-path", type=str, default="")
    p.add_argument("--module", type=str, default="")
    p.add_argument("--class-name", type=str, default="")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--metadata-name", type=str, default="features_manifest.csv")
    return p.parse_args()


def read_manifest(path: Path) -> list[Sample]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"id", "audio_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")

        out: list[Sample] = []
        for row in reader:
            out.append(
                Sample(
                    sample_id=str(row["id"]),
                    audio_path=Path(str(row["audio_path"])),
                    task=str(row.get("task", "default")),
                )
            )
    return out


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    wav, _ = librosa.load(path, sr=sample_rate, mono=True)
    if wav.size == 0:
        raise ValueError(f"empty audio: {path}")
    return wav.astype(np.float32, copy=False)


def collate_with_padding(wavs: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([w.shape[0] for w in wavs], dtype=torch.long)
    max_len = int(lengths.max())
    batch = torch.zeros((len(wavs), max_len), dtype=torch.float32)
    for i, wav in enumerate(wavs):
        t = torch.from_numpy(wav)
        batch[i, : t.shape[0]] = t
    return batch, lengths


def validate_embedding(vec: np.ndarray, expected_dim: int | None) -> int:
    if vec.size == 0:
        raise ValueError("empty embedding")
    if not np.isfinite(vec).all():
        raise ValueError("embedding contains NaN/Inf")
    if vec.ndim != 1:
        raise ValueError(f"embedding should be 1D after pooling, got shape={vec.shape}")
    if expected_dim is not None and vec.shape[0] != expected_dim:
        raise ValueError(f"embedding dim mismatch: got {vec.shape[0]}, expected {expected_dim}")
    return vec.shape[0]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = read_manifest(args.manifest)
    wrapper = WavJEPAInferenceWrapper(
        backend=args.backend,
        model_path=args.model_path,
        device=args.device,
        module=args.module,
        class_name=args.class_name,
    )

    expected_dim: int | None = None
    meta_rows: list[dict[str, Any]] = []

    for start in tqdm(range(0, len(samples), args.batch_size), desc="extract"):
        batch_samples = samples[start : start + args.batch_size]
        wavs = [load_audio(s.audio_path, args.sample_rate) for s in batch_samples]
        batch, lengths = collate_with_padding(wavs)
        batch = batch.to(wrapper.device)
        lengths = lengths.to(wrapper.device)

        emb = wrapper.encode(batch, lengths).detach().cpu().numpy().astype(np.float32, copy=False)

        for i, sample in enumerate(batch_samples):
            vec = emb[i].reshape(-1)
            expected_dim = validate_embedding(vec, expected_dim)

            out_path = args.output_dir / f"{sample.sample_id}.npy"
            np.save(out_path, vec)
            meta_rows.append(
                {
                    "id": sample.sample_id,
                    "task": sample.task,
                    "audio_path": str(sample.audio_path),
                    "feature_path": str(out_path),
                    "shape": str(tuple(vec.shape)),
                    "dtype": str(vec.dtype),
                }
            )

    meta_path = args.output_dir / args.metadata_name
    with meta_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "task", "audio_path", "feature_path", "shape", "dtype"])
        writer.writeheader()
        writer.writerows(meta_rows)

    summary = {
        "num_samples": len(meta_rows),
        "embedding_dim": expected_dim,
        "sample_rate": args.sample_rate,
        "backend": args.backend,
        "metadata_file": str(meta_path),
    }
    (args.output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
