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
        encoder_output: str,
    ) -> None:
        self.backend = backend
        self.encoder_output = encoder_output
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
        else:
            raise ValueError(f"unsupported backend: {backend}")

        self.model.eval()
        if hasattr(self.model, "to"):
            self.model.to(self.device)

    def _pick_from_dict(self, out: dict[str, Any]) -> torch.Tensor:
        encoder_priority = {
            "context": ("context_embeddings", "context_embedding", "context"),
            "target": ("target_embeddings", "target_embedding", "target"),
            "auto": (),
        }

        keys_to_try: tuple[str, ...] = encoder_priority[self.encoder_output] + (
            "embeddings",
            "embedding",
            "x",
            "features",
        )

        for key in keys_to_try:
            if key in out and isinstance(out[key], torch.Tensor):
                return out[key]

        tensor_candidates = [v for v in out.values() if isinstance(v, torch.Tensor)]
        if len(tensor_candidates) == 1:
            return tensor_candidates[0]

        available = sorted(out.keys())
        raise KeyError(
            "Could not choose embedding tensor from model output dict. "
            f"encoder_output={self.encoder_output}, available_keys={available}"
        )

    @torch.inference_mode()
    def encode(self, batch_audio: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if self.backend == "python" and hasattr(self.model, "encode"):
            out = self.model.encode(batch_audio, lengths)
        else:
            out = self.model(batch_audio, lengths)

        if isinstance(out, dict):
            out = self._pick_from_dict(out)
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

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=Path, help="CSV with columns: id,audio_path[,task]")
    src.add_argument("--audio-dir", type=Path, help="Directory of audio files (loaded recursively)")

    p.add_argument("--output-dir", type=Path, required=True, help="Output feature directory")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0, help="reserved for future use")
    p.add_argument("--backend", type=str, default="torchscript", choices=["torchscript", "python"])
    p.add_argument("--model-path", type=str, default="")
    p.add_argument("--module", type=str, default="")
    p.add_argument("--class-name", type=str, default="")
    p.add_argument("--encoder-output", type=str, default="context", choices=["context", "target", "auto"],
                   help="Select encoder output key from dict outputs")
    p.add_argument("--audio-exts", type=str, default=".wav,.flac,.mp3,.ogg,.m4a", help="comma-separated extensions")
    p.add_argument("--task", type=str, default="default", help="task name when using --audio-dir")
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


def discover_audio_samples(audio_dir: Path, extensions: set[str], task: str) -> list[Sample]:
    if not audio_dir.exists() or not audio_dir.is_dir():
        raise ValueError(f"invalid --audio-dir: {audio_dir}")

    samples: list[Sample] = []
    for path in sorted(audio_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        relative = path.relative_to(audio_dir)
        sample_id = str(relative.with_suffix("")).replace("/", "__")
        samples.append(Sample(sample_id=sample_id, audio_path=path, task=task))

    if not samples:
        raise ValueError(f"no audio files found in {audio_dir} (exts={sorted(extensions)})")
    return samples


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

    if args.manifest is not None:
        samples = read_manifest(args.manifest)
    else:
        exts = {ext.strip().lower() for ext in args.audio_exts.split(",") if ext.strip()}
        samples = discover_audio_samples(args.audio_dir, exts, args.task)

    wrapper = WavJEPAInferenceWrapper(
        backend=args.backend,
        model_path=args.model_path,
        device=args.device,
        module=args.module,
        class_name=args.class_name,
        encoder_output=args.encoder_output,
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
        "encoder_output": args.encoder_output,
        "metadata_file": str(meta_path),
    }
    (args.output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
