"""Post-training evaluation pipeline for the mnist_digit_uva EBM.

Run:  python main_evaluate.py
      N_GEN=10000 python main_evaluate.py
      CKPT=outputs/checkpoints/latest_checkpoint.pt python main_evaluate.py
      GEN_STEPS=1000 STEP_SIZE=10 NOISE=0.005 ANNEAL=1 python main_evaluate.py

Does NOT modify or retrain the generative model. It:
  1. loads the trained EBM checkpoint and generates N samples (via the repo's Sampler),
  2. loads the MNIST test set (with labels),
  3. loads a pretrained MNIST classifier (trained once and cached; see evaluation/classifier.py),
  4. computes FID (classifier features), nearest-neighbour stats, and digit distribution,
  5. writes all seven figures, digit_counts.csv, and evaluation_results.json.

Everything is driven by env vars with sensible defaults, so it runs unattended.
"""
from __future__ import annotations

import os

import torch
import torch.utils.data as tud
from torchvision.datasets import MNIST
from torchvision import transforms

# --- the trained generative model lives in the parent package ---
from model import CNNModel
from sampler import Sampler
import utils as repo_utils   # the EBM repo's utils (checkpoint loading)

# --- the evaluation package ---
from evaluation import utils, classifier, fid, nearest_neighbor, metrics, plots


# --------------------------------------------------------------------------- config (env)
CKPT = os.getenv("CKPT", "outputs/checkpoints/best_checkpoint.pt")
N_GEN = int(os.getenv("N_GEN", "10000"))
GEN_STEPS = int(os.getenv("GEN_STEPS", "256"))
STEP_SIZE = float(os.getenv("STEP_SIZE", "10"))
NOISE = float(os.getenv("NOISE", "0.005"))
ANNEAL = os.getenv("ANNEAL") == "1"
NOISE_END = float(os.getenv("NOISE_END", str(NOISE / 50.0)))
GEN_BATCH = int(os.getenv("GEN_BATCH", "1000"))     # generate in chunks to bound memory
SEED = int(os.getenv("SEED", "0"))
DATA_ROOT = os.getenv("DATA_ROOT", "./data")

OUT_JSON = "evaluation_results.json"
OUT_CSV = "digit_counts.csv"
PLOT_DIR = "evaluation_plots"


# --------------------------------------------------------------------------- steps
def load_generator(device: torch.device):
    """Rebuild the trained EBM from its checkpoint (same recipe as inference.py)."""
    peek = torch.load(CKPT, map_location=device)
    saved_cfg = peek.get("config", {})
    model = CNNModel(
        hidden_features=saved_cfg.get("hidden_features", 32),
        activation=saved_cfg.get("activation", "swish"),
        grad_clamp=saved_cfg.get("grad_clamp", 0.03),
    ).to(device)
    ckpt = repo_utils.load_checkpoint(CKPT, model, optimizer=None, map_location=device)
    model.eval()
    cfg = ckpt.get("config", {})
    img_shape = cfg.get("img_shape", (1, 28, 28))
    print(f"[gen] loaded {CKPT} (epoch {ckpt.get('epoch', '?')}, "
          f"digit {cfg.get('digit_class', '?')})")
    return model, img_shape


@torch.no_grad()
def generate_samples(model, img_shape, device: torch.device) -> torch.Tensor:
    """Generate N_GEN images by Langevin sampling, in GEN_BATCH-sized chunks."""
    print(f"[gen] generating {N_GEN} samples ({GEN_STEPS} steps, step_size={STEP_SIZE}, "
          f"noise={NOISE}, anneal={ANNEAL})...")
    chunks = []
    remaining = N_GEN
    while remaining > 0:
        b = min(GEN_BATCH, remaining)
        start = torch.rand((b,) + img_shape, device=device) * 2 - 1
        s = Sampler.generate_samples(
            model, start, steps=GEN_STEPS, step_size=STEP_SIZE,
            noise=NOISE, anneal=ANNEAL, noise_end=NOISE_END,
        )
        chunks.append(s.cpu())
        remaining -= b
        print(f"[gen]   {N_GEN - remaining}/{N_GEN}")
    return torch.cat(chunks, dim=0)


def load_mnist_test(device: torch.device):
    """MNIST test set in EBM [-1,1] space, plus integer labels. (10,000 images.)"""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),    # -> [-1, 1], matching the EBM
    ])
    test = MNIST(root=DATA_ROOT, train=False, transform=tf, download=True)
    loader = tud.DataLoader(test, batch_size=1024, shuffle=False)
    imgs, labels = [], []
    for x, y in loader:
        imgs.append(x); labels.append(y)
    real_images = torch.cat(imgs, dim=0)          # (10000, 1, 28, 28) in [-1,1]
    real_labels = torch.cat(labels, dim=0).numpy()
    print(f"[data] MNIST test: {real_images.shape[0]} images")
    return real_images, real_labels


# --------------------------------------------------------------------------- main
def main() -> None:
    utils.set_seed(SEED)
    device = utils.get_device()
    utils.ensure_dir(PLOT_DIR)
    print(f"device: {device}")

    # 1-2. model + samples + real data
    model, img_shape = load_generator(device)
    gen_images = generate_samples(model, img_shape, device)
    real_images, real_labels = load_mnist_test(device)

    # 3. pretrained classifier (train-once-cache), used for features + labels
    clf = classifier.load_or_train_classifier(device, data_root=DATA_ROOT)

    # 4. FID (classifier features)
    print("[fid] computing FID...")
    fid_value = fid.compute_fid(clf, real_images, gen_images, device)
    print(f"FID = {fid_value:.2f}")

    # 5. nearest-neighbour analysis
    print("[nn] computing nearest neighbours...")
    nn_idx, nn_dist = nearest_neighbor.nearest_neighbors(gen_images, real_images, device)
    nn_stats = nearest_neighbor.distance_statistics(nn_dist)
    print(f"[nn] mean={nn_stats['mean']:.3f} median={nn_stats['median']:.3f} "
          f"min={nn_stats['min']:.3f} max={nn_stats['max']:.3f} std={nn_stats['std']:.3f}")

    # 6. digit distribution
    print("[dist] classifying generated samples...")
    gen_labels = metrics.classify_generated(clf, gen_images, device)
    metrics.digit_distribution(gen_labels, real_labels, OUT_CSV)

    # 7. plots
    print("[plots] rendering figures...")
    plots.generated_samples_grid(gen_images, PLOT_DIR)
    plots.digit_histogram(gen_labels, real_labels, PLOT_DIR)
    plots.nearest_neighbor_examples(gen_images, real_images, nn_idx, nn_dist, PLOT_DIR)
    plots.nearest_neighbor_distance_histogram(nn_dist, PLOT_DIR)
    plots.confusion_heatmap(gen_labels, PLOT_DIR)
    plots.mean_digit_per_class(gen_images, gen_labels, PLOT_DIR)
    plots.closest_distance_per_digit(gen_labels, nn_dist, PLOT_DIR)
    print(f"[plots] wrote 7 figures -> {PLOT_DIR}/")

    # 8. metrics
    results = metrics.build_results(N_GEN, fid_value, nn_stats, gen_labels)
    metrics.save_results(results, OUT_JSON)

    print("\n=== evaluation complete ===")
    print(f"FID = {fid_value:.2f}")
    print(f"results -> {OUT_JSON}, {OUT_CSV}, {PLOT_DIR}/")


if __name__ == "__main__":
    main()
