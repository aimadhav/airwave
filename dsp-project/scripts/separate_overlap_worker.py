"""Offline CPU separation worker. Only trusted, hash-checked local model YAML."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main() -> None:
    """Resample with antialiasing; retain shared gain and exact session length."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--model-rate", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    import torch
    from scipy.signal import resample_poly
    from math import gcd
    from hyperpyyaml import load_hyperpyyaml
    from speechbrain.inference.separation import SepformerSeparation

    torch.set_num_threads(args.threads)
    manifest = json.loads((args.model / "manifest.json").read_text(encoding="utf-8"))
    if args.model_rate != manifest["sample_rate"]:
        raise ValueError("Separator sample rate differs from installed manifest")
    for filename, expected in manifest["sha256"].items():
        path = args.model / filename
        if path.parent.resolve() != args.model.resolve():
            raise ValueError("Invalid model manifest path")
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != expected:
                raise ValueError(f"Model integrity check failed: {filename}")
    # Load the inspected architecture and state dicts locally, bypassing all HF
    # fetching, custom.py execution, symlink handling and pickle object loading.
    with (args.model / "hyperparams.yaml").open(encoding="utf-8") as source:
        hparams = load_hyperpyyaml(source)
    if hparams["sample_rate"] != args.model_rate or hparams["num_spks"] != 2:
        raise ValueError("Unsupported separator contract")
    for name, module in hparams["modules"].items():
        module.load_state_dict(torch.load(args.model / f"{name}.ckpt", map_location="cpu", weights_only=True))
    model = SepformerSeparation(modules=hparams["modules"], hparams=hparams, run_opts={"device": "cpu"})
    samples = np.load(args.input, allow_pickle=False)
    if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
        raise ValueError("Invalid separator audio")
    factor = gcd(args.sample_rate, args.model_rate)
    down = resample_poly(samples, args.model_rate // factor, args.sample_rate // factor).astype(np.float32)
    with torch.inference_mode():
        separated = model.separate_batch(torch.from_numpy(down).unsqueeze(0))[0].cpu().numpy()
    streams: dict[str, np.ndarray] = {}
    for i in range(2):
        track = resample_poly(separated[:, i], args.sample_rate // factor, args.model_rate // factor)
        track = np.pad(track, (0, max(0, len(samples) - len(track))))[:len(samples)]
        streams[f"track_{i}"] = track.astype(np.float32)
    np.savez(args.output, **streams)


if __name__ == "__main__":
    main()
