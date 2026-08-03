"""Train a (single-digit or full) MNIST EBM -- UVA's exact algorithm, on TorchEBM.

Run:  python train.py

WHAT THIS IS
------------
The UvA DL tutorial 8 energy model, reproduced faithfully but built on TorchEBM's
BaseModel (so the library computes grad_x E) with a hand-written Langevin loop and CD
loss, because those two pieces need UVA's exact behaviour:

  - MODEL     : UVA's Swish CNN (28->16->8->4->2 funnel), as a torchebm BaseModel.
  - GRADIENT  : torchebm BaseModel.gradient(), with UVA's [-0.03, 0.03] clamp injected.
  - SAMPLER   : UVA's Langevin ordering (noise FIRST, then grad at the noised point)
                + persistent replay buffer. See sampler.py for why this is custom.
  - LOSS      : contrastive divergence, hand-written (see cd_step below for the sign).

SIGN CONVENTION: the model returns the ENERGY E (TorchEBM's convention, NOT the
notebook's score f = -E). Lower energy = more data-like. Because every component agrees
on this, inference never needs a NEGATE flag.

Everything configurable lives in the CONFIG block below.
"""
from __future__ import annotations

import os
import time

import torch
import torch.utils.data as data
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm

import utils
from model import CNNModel
from sampler import Sampler

# --------------------------------------------------------------------------- CONFIG
# Defaults reproduce the UVA notebook. Change experiments by editing ONLY this dict.
CONFIG = {
    # --- data ---
    "digit_class": 3,             # None = all digits | 0..9 = train on ONLY that class
    "batch_size": 128,
    "val_fraction": 0.1,
    "data_root": "./data",
    "img_shape": (1, 28, 28),

    # --- model (UVA defaults) ---
    "hidden_features": 32,        # channel ladder: hidden//2 -> hidden -> hidden*2
    "activation": "swish",        # swish (UVA) | silu | relu | gelu | softplus
    "grad_clamp": 0.03,           # UVA's element-wise cap on grad_x E (None = off)

    # --- optimization (UVA defaults) ---
    "epochs": 60,
    "lr": 1e-4,
    "beta1": 0.0,                 # Adam momentum OFF: the EBM's target moves as theta
                                  # changes, so old-gradient momentum is stale.
    "lr_gamma": 0.97,             # StepLR: multiply lr by this every epoch
    "grad_clip": 0.1,             # clip grad_THETA norm (weight updates).
                                  # NOTE: different from grad_clamp (grad_x E, sampling).
    "seed": 42,

    # --- contrastive divergence ---
    "cd_steps": 60,               # Langevin steps per training batch (k)
    "cd_step_size": 10,
    "alpha": 0.1,                 # weight of the E^2 regularizer (bounds the energy scale)
    "real_jitter": 0.005,         # noise added to REAL images before the loss (UVA)
    "buffer_size": 8192,
    "new_sample_ratio": 0.05,     # 5% fresh noise / 95% resumed chains
    "langevin_noise": 0.005,      # per-step Langevin noise std

    # --- generation ---
    "gen_steps": 256,             # Langevin steps for GENERATION (longer than training)
    "gen_step_size": 10,
    "n_gen_samples": 64,
    "sample_every": 5,            # save a sample grid every N epochs

    # --- optional noise annealing (OFF = exact UVA behaviour) ---
    # High noise early = chains explore and spread out (fixes identical-sample collapse);
    # low noise late = chains settle cleanly (fixes speckle). Applies to GENERATION.
    "anneal": False,
    "anneal_noise_start": 0.05,   # used only when anneal=True
    "anneal_noise_end": 0.001,

    # --- output ---
    "out_dir": "outputs",
}


# --------------------------------------------------------------------------- data
class _SingleClass(data.Dataset):
    """Expose ONLY the samples whose label == digit_class.

    Indices are precomputed once (no per-item scan), so single-mode training is just as
    fast as the full set.
    """
    def __init__(self, base: data.Dataset, digit_class: int):
        self.base = base
        targets = base.targets
        if not torch.is_tensor(targets):
            targets = torch.tensor(targets)
        self.indices = (targets == digit_class).nonzero(as_tuple=True)[0].tolist()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.base[self.indices[i]]


def build_dataloaders(config: dict) -> tuple[data.DataLoader, data.DataLoader]:
    """MNIST -> (train_loader, val_loader), normalized to [-1, 1].

    The [-1, 1] range MUST match the sampler's clamp and the noise-init range, so real
    and generated images live in the same space.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                      # [0, 1]
        transforms.Normalize((0.5,), (0.5,)),       # -> [-1, 1]
    ])
    full_train = MNIST(root=config["data_root"], train=True,
                       transform=transform, download=True)

    digit = config["digit_class"]
    if digit is not None:
        full_train = _SingleClass(full_train, digit)
        print(f"filtered to digit {digit}: {len(full_train)} training images")

    val_size = int(len(full_train) * config["val_fraction"])
    train_size = len(full_train) - val_size
    gen = torch.Generator().manual_seed(config["seed"])
    train_set, val_set = data.random_split(full_train, [train_size, val_size], generator=gen)

    train_loader = data.DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True,
        drop_last=True, num_workers=4, pin_memory=True,
    )
    val_loader = data.DataLoader(
        val_set, batch_size=config["batch_size"], shuffle=False,
        drop_last=False, num_workers=4, pin_memory=True,
    )
    return train_loader, val_loader


# --------------------------------------------------------------------------- CD step
def cd_step(model: CNNModel, sampler: Sampler, real_imgs: torch.Tensor, config: dict):
    """One contrastive-divergence step. Returns (loss, E_real_mean, E_fake_mean).

    THE SIGN (stare at this once):
        Our model returns the ENERGY E. We want real images to have LOW energy and the
        model's own samples (negatives) to have HIGH energy, so:

            cdiv = E(real).mean() - E(fake).mean()      <-- minimizing pushes E_real DOWN
                                                            and E_fake UP.

        The UVA notebook writes `fake_out.mean() - real_out.mean()` because ITS network
        returns the score f = -E. Same objective, opposite sign convention.

    The alpha * (E_real^2 + E_fake^2) regularizer bounds the ENERGY SCALE. Without it the
    CD term only constrains the DIFFERENCE, so the absolute level is free to drift.
    """
    # Jitter real images so the model doesn't overfit to perfectly clean inputs (UVA).
    if config["real_jitter"]:
        real_imgs = real_imgs + torch.randn_like(real_imgs) * config["real_jitter"]
        real_imgs = real_imgs.clamp(-1.0, 1.0)

    # NEGATIVES: Langevin from the replay buffer (grad_x phase, theta frozen inside).
    fake_imgs = sampler.sample_new_exmps(
        steps=config["cd_steps"], step_size=config["cd_step_size"],
    )

    # One batched forward over [real; fake], then split -> identical model state for both.
    inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
    real_energy, fake_energy = model(inp_imgs).chunk(2, dim=0)

    reg_loss = config["alpha"] * (real_energy ** 2 + fake_energy ** 2).mean()
    cdiv_loss = real_energy.mean() - fake_energy.mean()
    loss = reg_loss + cdiv_loss
    return loss, real_energy.mean().item(), fake_energy.mean().item()


# --------------------------------------------------------------------------- generation
def generate(model: CNNModel, config: dict, device: torch.device,
             n: int, steps: int) -> torch.Tensor:
    """Generate `n` images by rolling noise downhill on the energy for `steps` steps."""
    start = torch.rand((n,) + config["img_shape"], device=device) * 2 - 1
    noise = config["anneal_noise_start"] if config["anneal"] else config["langevin_noise"]
    return Sampler.generate_samples(
        model, start, steps=steps, step_size=config["gen_step_size"],
        noise=noise, anneal=config["anneal"], noise_end=config["anneal_noise_end"],
    )


def energy_of(model: CNNModel, x: torch.Tensor) -> float:
    """Mean scalar energy of a batch (no grad)."""
    with torch.no_grad():
        return model(x).mean().item()


# --------------------------------------------------------------------------- main
def main() -> None:
    config = CONFIG
    utils.set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  |  digit: {config['digit_class']}  |  epochs: {config['epochs']}"
          f"  |  activation: {config['activation']}  |  anneal: {config['anneal']}")

    out = config["out_dir"]
    ckpt_dir = os.path.join(out, "checkpoints")
    sample_dir = os.path.join(out, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    train_loader, val_loader = build_dataloaders(config)
    model = CNNModel(
        hidden_features=config["hidden_features"],
        activation=config["activation"],
        grad_clamp=config["grad_clamp"],
    ).to(device)
    sampler = Sampler(
        model, img_shape=config["img_shape"], sample_size=config["batch_size"],
        device=device, max_len=config["buffer_size"],
        noise=config["langevin_noise"], new_sample_ratio=config["new_sample_ratio"],
    )

    # beta1=0 (momentum off) + StepLR decay -- both UVA defaults.
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                 betas=(config["beta1"], 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1,
                                                gamma=config["lr_gamma"])

    # --- BEFORE training: baseline samples (should look like noise) ---
    baseline = generate(model, config, device, config["n_gen_samples"], config["gen_steps"])
    utils.save_sample_grid(baseline, os.path.join(sample_dir, "epoch_000_before.png"))
    print("saved baseline samples -> epoch_000_before.png")

    history = []
    best_loss = float("inf")
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        t0 = time.time()
        epoch_loss = e_real_sum = e_fake_sum = 0.0
        n_batches = 0

        for real_imgs, _labels in tqdm(train_loader, desc=f"epoch {epoch}/{config['epochs']}",
                                       leave=False):
            real_imgs = real_imgs.to(device)
            loss, e_real, e_fake = cd_step(model, sampler, real_imgs, config)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()

            epoch_loss += loss.item()
            e_real_sum += e_real
            e_fake_sum += e_fake
            n_batches += 1

        scheduler.step()
        n = max(n_batches, 1)
        epoch_loss /= n
        e_real_avg = e_real_sum / n
        e_fake_avg = e_fake_sum / n
        epoch_time = time.time() - t0

        history.append({
            "epoch": epoch, "loss": epoch_loss,
            "e_real": e_real_avg, "e_fake": e_fake_avg, "time_sec": epoch_time,
        })
        print(f"epoch {epoch:3d}  loss {epoch_loss:+.4f}  E_real {e_real_avg:+.3f}  "
              f"E_fake {e_fake_avg:+.3f}  gap {e_fake_avg - e_real_avg:+.3f}  [{epoch_time:.1f}s]")

        utils.save_checkpoint(os.path.join(ckpt_dir, "latest_checkpoint.pt"),
                              model, optimizer, epoch, epoch_loss, config)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            utils.save_checkpoint(os.path.join(ckpt_dir, "best_checkpoint.pt"),
                                  model, optimizer, epoch, epoch_loss, config)

        if epoch % config["sample_every"] == 0:
            samples = generate(model, config, device, config["n_gen_samples"], config["gen_steps"])
            utils.save_sample_grid(samples, os.path.join(sample_dir, f"epoch_{epoch:03d}.png"))

    # --- AFTER training: final samples + metrics + plots ---
    final = generate(model, config, device, config["n_gen_samples"], config["gen_steps"])
    utils.save_sample_grid(final, os.path.join(sample_dir, "final_samples.png"))
    utils.save_metrics(history, out)
    utils.plot_curves(history, os.path.join(out, "plots"))

    # --- sign check: real images should have LOWER energy than pure noise ---
    real_batch = next(iter(val_loader))[0].to(device)
    noise = torch.rand(256, *config["img_shape"], device=device) * 2 - 1
    e_real = energy_of(model, real_batch)
    e_noise = energy_of(model, noise)
    sign_ok = e_real < e_noise
    sign_msg = (
        f"E(real)  = {e_real:+.3f}\nE(noise) = {e_noise:+.3f}\n"
        f"real < noise ? {sign_ok}\n"
        + ("OK: low energy = data. inference.py works as-is.\n" if sign_ok
           else "UNEXPECTED: real should score LOWER than noise -- check convergence.\n")
    )
    with open(os.path.join(out, "sign_check.txt"), "w") as f:
        f.write(sign_msg)
    print("\n=== SIGN CHECK ===\n" + sign_msg)
    print(f"done. outputs in ./{out}/")


if __name__ == "__main__":
    main()
