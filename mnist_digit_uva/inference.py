"""Generate MNIST digits from a trained UVA-on-TorchEBM checkpoint.

Run:  python inference.py
      N_STEPS=1000 python inference.py                  # longer chain
      ANNEAL=1 python inference.py                      # annealed noise (high -> low)
      ANNEAL=1 NOISE=0.1 NOISE_END=0.001 N_STEPS=1000 python inference.py
      STEP_SIZE=1.0 NOISE=0.05 python inference.py      # gentler drift, more exploration
      CKPT=outputs/checkpoints/latest_checkpoint.pt python inference.py

NO `NEGATE` FLAG: the model returns the ENERGY E (TorchEBM's convention) and the sampler
descends grad_x E, so the sign is consistent by construction. A trained model always
generates without a flip. (train.py still writes sign_check.txt as a health readout.)

IF ALL YOUR SAMPLES LOOK IDENTICAL (mode collapse):
    the chain is behaving like deterministic gradient descent -- the drift term
    (step_size * grad) is swamping the noise term, so every chain finds the SAME
    minimum. Raise NOISE and/or lower STEP_SIZE, or turn on ANNEAL=1. Diversity first,
    sharpness second.
"""
from __future__ import annotations

import os

import torch

import utils
from model import CNNModel
from sampler import Sampler

# --------------------------------------------------------------------------- config
CKPT = os.getenv("CKPT", "outputs/checkpoints/latest_checkpoint.pt")
N_SAMPLES = int(os.getenv("N_SAMPLES", "64"))
N_STEPS = int(os.getenv("N_STEPS", "256"))
STEP_SIZE = float(os.getenv("STEP_SIZE", "10"))
NOISE = float(os.getenv("NOISE", "0.005"))
ANNEAL = os.getenv("ANNEAL") == "1"
NOISE_END = float(os.getenv("NOISE_END", str(NOISE / 50.0)))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load checkpoint; peek at the saved config so we rebuild the SAME architecture ---
    peek = torch.load(CKPT, map_location=device)
    saved_cfg = peek.get("config", {})
    model = CNNModel(
        hidden_features=saved_cfg.get("hidden_features", 32),
        activation=saved_cfg.get("activation", "swish"),
        grad_clamp=saved_cfg.get("grad_clamp", 0.03),
    ).to(device)
    ckpt = utils.load_checkpoint(CKPT, model, optimizer=None, map_location=device)
    model.eval()
    config = ckpt.get("config", {})
    print(f"loaded {CKPT}  (trained {ckpt.get('epoch', '?')} epochs, "
          f"loss {ckpt.get('loss', float('nan')):+.4f}, digit {config.get('digit_class', '?')})")

    img_shape = config.get("img_shape", (1, 28, 28))

    # --- sign sanity: real data should score LOWER than noise ---
    with torch.no_grad():
        noise_batch = torch.rand(256, *img_shape, device=device) * 2 - 1
        print(f"E(random noise) = {model(noise_batch).mean().item():+.3f}  "
              f"(a real digit should score LOWER)")

    print(f"generating {N_SAMPLES} samples: {N_STEPS} steps, step_size={STEP_SIZE}, "
          f"noise={NOISE}, anneal={ANNEAL}" + (f" -> {NOISE_END}" if ANNEAL else ""))

    start = torch.rand((N_SAMPLES,) + img_shape, device=device) * 2 - 1
    samples = Sampler.generate_samples(
        model, start, steps=N_STEPS, step_size=STEP_SIZE,
        noise=NOISE, anneal=ANNEAL, noise_end=NOISE_END,
    )

    # Diversity readout: mean pairwise distance between samples. Near 0 => mode collapse
    # (every image identical). This is the number to watch if the grid looks repetitive.
    flat = samples.flatten(1)
    diversity = torch.cdist(flat, flat).mean().item()
    print(f"sample diversity (mean pairwise L2) = {diversity:.3f}  "
          f"(near 0 => all samples identical; raise NOISE or use ANNEAL=1)")

    out_path = os.path.join(config.get("out_dir", "outputs"), "samples", "inference_samples.png")
    utils.save_sample_grid(samples, out_path)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
