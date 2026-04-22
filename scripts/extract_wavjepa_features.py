#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import pkgutil
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
        hf_model_id: str,
        hf_filename: str,
        sample_rate: int,
    ) -> None:
        self.backend = self._resolve_backend(backend, model_path, module, class_name, hf_model_id)
        self.encoder_output = encoder_output
        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self.feature_extractor = None
        model_path = self._resolve_model_path(
            model_path=model_path,
            hf_model_id=hf_model_id,
            hf_filename=hf_filename,
        )

        if self.backend == "torchscript":
            if not model_path:
                raise ValueError("--model-path is required when --backend=torchscript")
            self.model = torch.jit.load(model_path, map_location=self.device)
        elif self.backend == "python":
            cls = self._resolve_python_class(module, class_name)
            self.model = cls(model_path)
        elif self.backend == "python-ckpt":
            if not model_path:
                raise ValueError("--model-path is required when --backend=python-ckpt")
            cls = self._resolve_python_class(module, class_name)
            self.model = self._load_with_lightning(cls, model_path)
            if self.model is None:
                self.model = self._build_python_model(cls, model_path)
                self._load_checkpoint_into_model(self.model, model_path)
        elif self.backend == "python-safetensors":
            if not model_path:
                raise ValueError("--model-path (or --hf-model-id) is required when --backend=python-safetensors")
            cls = self._resolve_python_class(module, class_name)
            self.model = self._build_python_model(cls, model_path)
            self._load_safetensors_into_model(self.model, model_path)
        elif self.backend == "wavjepa-hf":
            repo_or_path = hf_model_id or model_path
            if not repo_or_path:
                raise ValueError("--hf-model-id (or --model-path as local hf dir) is required when --backend=wavjepa-hf")
            self.model, self.feature_extractor = self._load_hf_wavjepa(repo_or_path, self.device)
        else:
            raise ValueError(f"unsupported backend: {self.backend}")

        self.model.eval()
        if hasattr(self.model, "to"):
            self.model.to(self.device)

    @staticmethod
    def _resolve_backend(backend: str, model_path: str, module: str, class_name: str, hf_model_id: str) -> str:
        if backend != "auto":
            return backend
        if hf_model_id and not model_path:
            return "wavjepa-hf"
        suffix = Path(model_path).suffix.lower()
        if suffix in {".ts", ".torchscript"}:
            return "torchscript"
        if suffix == ".safetensors":
            return "python-safetensors"
        if suffix in {".ckpt", ".pt", ".pth"}:
            return "python-ckpt"
        if module and class_name:
            return "python"
        return "torchscript"

    @staticmethod
    def _load_hf_wavjepa(repo_or_path: str, device: torch.device) -> tuple[Any, Any]:
        try:
            from transformers import AutoFeatureExtractor, AutoModel
        except Exception as exc:
            raise RuntimeError(
                "WavJEPA HF backend requires `transformers`. Install it first."
            ) from exc

        model = AutoModel.from_pretrained(repo_or_path, trust_remote_code=True).to(device)
        extractor = AutoFeatureExtractor.from_pretrained(repo_or_path, trust_remote_code=True)
        return model, extractor

    def _resolve_python_class(self, module: str, class_name: str) -> type:
        if module and class_name:
            mod = importlib.import_module(module)
            return getattr(mod, class_name)

        discovered = self._discover_wavjepa_class()
        if discovered is not None:
            return discovered

        raise ValueError(
            "Could not resolve model class automatically. "
            "Pass --module/--class-name or install official WavJEPA code (package `wavjepa`/`sjepa`) "
            "in the current Python environment."
        )

    @staticmethod
    def _discover_wavjepa_class() -> type | None:
        candidate_roots = ("wavjepa", "sjepa")
        visited_modules: set[str] = set()
        scored_classes: list[tuple[int, type]] = []

        for root in candidate_roots:
            try:
                root_mod = importlib.import_module(root)
            except Exception:
                continue

            modules = [root_mod]
            pkg_path = getattr(root_mod, "__path__", None)
            if pkg_path is not None:
                for modinfo in pkgutil.walk_packages(pkg_path, prefix=f"{root}."):
                    try:
                        modules.append(importlib.import_module(modinfo.name))
                    except Exception:
                        continue

            for mod in modules:
                mod_name = getattr(mod, "__name__", "")
                if not mod_name or mod_name in visited_modules:
                    continue
                visited_modules.add(mod_name)

                for _, cls in inspect.getmembers(mod, inspect.isclass):
                    if cls.__module__ != mod_name:
                        continue
                    if not issubclass(cls, torch.nn.Module):
                        continue
                    name = cls.__name__.lower()
                    score = 0
                    if "jepa" in name:
                        score += 10
                    if "wav" in name:
                        score += 5
                    scored_classes.append((score, cls))

        if not scored_classes:
            return None
        scored_classes.sort(key=lambda item: item[0], reverse=True)
        return scored_classes[0][1]

    @staticmethod
    def _resolve_model_path(model_path: str, hf_model_id: str, hf_filename: str) -> str:
        if model_path:
            return model_path
        if not hf_model_id:
            return model_path
        filename = hf_filename or "model.safetensors"
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:
            raise RuntimeError(
                "Downloading from Hugging Face requires `huggingface_hub`. "
                "Install it first or pass --model-path with a local file."
            ) from exc
        return hf_hub_download(repo_id=hf_model_id, filename=filename)

    @staticmethod
    def _build_python_model(cls: type, model_path: str) -> Any:
        init_sig = inspect.signature(cls.__init__)
        param_names = [name for name in init_sig.parameters if name != "self"]
        if "model_path" in param_names:
            return cls(model_path=model_path)
        if param_names:
            return cls(model_path)
        return cls()

    def _load_with_lightning(self, cls: type, model_path: str) -> Any | None:
        loader = getattr(cls, "load_from_checkpoint", None)
        if loader is None:
            return None
        for kwargs in (
            {"checkpoint_path": model_path, "map_location": self.device, "strict": False},
            {"checkpoint_path": model_path, "map_location": self.device},
            {"checkpoint_path": model_path},
        ):
            try:
                return loader(**kwargs)
            except TypeError:
                continue
            except Exception:
                break
        return None

    @staticmethod
    def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
        if isinstance(payload, dict):
            for key in ("state_dict", "model_state_dict"):
                if key in payload and isinstance(payload[key], dict):
                    return payload[key]
            if all(isinstance(k, str) for k in payload.keys()):
                return payload
        raise ValueError("Could not extract state_dict from checkpoint")

    def _load_checkpoint_into_model(self, model: Any, model_path: str) -> None:
        payload = torch.load(model_path, map_location=self.device)
        state_dict = self._extract_state_dict(payload)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if len(missing) == 0:
            return
        strip_prefixes = ("model.", "module.", "net.", "backbone.")
        stripped_state = {
            next((k[len(p):] for p in strip_prefixes if k.startswith(p)), k): v
            for k, v in state_dict.items()
        }
        missing_retry, unexpected_retry = model.load_state_dict(stripped_state, strict=False)
        if len(missing_retry) > len(missing) and len(unexpected_retry) >= len(unexpected):
            raise RuntimeError(
                "checkpoint load failed: excessive missing keys after load_state_dict. "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

    def _load_safetensors_into_model(self, model: Any, model_path: str) -> None:
        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError(
                "Loading .safetensors requires `safetensors`. Install it first."
            ) from exc

        state_dict = load_file(model_path, device=str(self.device))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if len(missing) == 0:
            return
        strip_prefixes = ("model.", "module.", "net.", "backbone.")
        stripped_state = {
            next((k[len(p):] for p in strip_prefixes if k.startswith(p)), k): v
            for k, v in state_dict.items()
        }
        missing_retry, unexpected_retry = model.load_state_dict(stripped_state, strict=False)
        if len(missing_retry) > len(missing) and len(unexpected_retry) >= len(unexpected):
            raise RuntimeError(
                "safetensors load failed: excessive missing keys after load_state_dict. "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

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
            "last_hidden_state",
            "pooler_output",
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
        if self.backend == "wavjepa-hf":
            if self.feature_extractor is None:
                raise RuntimeError("feature_extractor is not initialized for wavjepa-hf backend")

            extracted = self.feature_extractor(
                batch_audio.detach().cpu().numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            input_values = extracted["input_values"].to(self.device)
            out = self.model(input_values)
        elif self.backend == "python-ckpt" and hasattr(self.model, "extract_audio") and hasattr(self.model, "feature_norms"):
            local = self.model.extract_audio(batch_audio)
            local = self.model.feature_norms(local)
            if hasattr(self.model, "post_extraction_mapper") and self.model.post_extraction_mapper is not None:
                local = self.model.post_extraction_mapper(local)
            if hasattr(self.model, "pos_encoding_encoder"):
                local = local + self.model.pos_encoding_encoder[:, : local.shape[1], :].to(local.device)

            mask = torch.zeros((local.shape[0], local.shape[1]), dtype=torch.bool, device=local.device)
            if hasattr(self.model, "encoder_forward"):
                out = self.model.encoder_forward(local, src_key_padding_mask=mask)
            elif hasattr(self.model, "encoder"):
                out = self.model.encoder(local, src_key_padding_mask=mask)
            else:
                out = local
        else:
            call_attempts = []
            if hasattr(self.model, "encode"):
                call_attempts.extend([
                    lambda: self.model.encode(batch_audio, lengths),
                    lambda: self.model.encode(batch_audio),
                ])
            call_attempts.extend([
                lambda: self.model(batch_audio, lengths),
                lambda: self.model(batch_audio),
            ])

            out = None
            last_exc: Exception | None = None
            for fn in call_attempts:
                try:
                    out = fn()
                    break
                except TypeError as exc:
                    last_exc = exc
                    continue
            if out is None:
                raise RuntimeError(f"model forward invocation failed: {last_exc}")

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
    p.add_argument("--audio-dir", type=Path, required=True, help="Directory of audio files (loaded recursively)")

    p.add_argument("--output-dir", type=Path, required=True, help="Output feature directory")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0, help="reserved for future use")
    p.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "torchscript", "python", "python-ckpt", "python-safetensors", "wavjepa-hf"],
    )
    p.add_argument("--model-path", type=str, default="")
    p.add_argument("--hf-model-id", type=str, default="", help="Hugging Face model repo id")
    p.add_argument("--hf-filename", type=str, default="model.safetensors", help="Filename in HF repo")
    p.add_argument("--module", type=str, default="",
                   help="Python module for model class. Optional when official wavjepa/sjepa package is installed.")
    p.add_argument("--class-name", type=str, default="",
                   help="Model class name. Optional when official wavjepa/sjepa package is installed.")
    p.add_argument("--encoder-output", type=str, default="context", choices=["context", "target", "auto"],
                   help="Select encoder output key from dict outputs")
    p.add_argument("--audio-exts", type=str, default=".wav,.flac,.mp3,.ogg,.m4a", help="comma-separated extensions")
    p.add_argument("--task", type=str, default="default", help="task name when using --audio-dir")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--metadata-name", type=str, default="features_manifest.csv")
    return p.parse_args()


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

    exts = {ext.strip().lower() for ext in args.audio_exts.split(",") if ext.strip()}
    samples = discover_audio_samples(args.audio_dir, exts, args.task)

    wrapper = WavJEPAInferenceWrapper(
        backend=args.backend,
        model_path=args.model_path,
        device=args.device,
        module=args.module,
        class_name=args.class_name,
        encoder_output=args.encoder_output,
        hf_model_id=args.hf_model_id,
        hf_filename=args.hf_filename,
        sample_rate=args.sample_rate,
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
