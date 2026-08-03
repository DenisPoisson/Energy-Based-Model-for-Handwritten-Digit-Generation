"""Langevin MCMC sampler + persistent replay buffer -- UVA's exact algorithm, driven by
TorchEBM's model gradient. Plus an OPTIONAL noise-annealing schedule (off by default,
so the default path is bit-faithful to UVA).

WHY A CUSTOM LOOP INSTEAD OF torchebm.samplers.LangevinDynamics
---------------------------------------------------------------
The ORDERING of noise and gradient differs, and it matters:

    UVA (this file):     perturb x with noise  ->  evaluate grad_x E AT THE NOISED x  ->  step
    LangevinDynamics:    evaluate drift at x   ->  integrator adds noise as part of the update

Both are valid discretizations of the same SDE, but UVA's injects stochasticity INTO the
gradient evaluation, which decorrelates parallel chains far more strongly. With constant
tiny noise and a large step size, the stock ordering can behave like deterministic
gradient descent -- every chain rolls into the SAME minimum, producing a grid of near
identical samples (mode collapse). LangevinDynamics exposes no hook to reorder this
(it happens inside integrator.step()), so we write the 10-line loop ourselves.

TorchEBM still does the real work: we call `model.gradient(x)`, which is BaseModel's
autograd implementation PLUS our subclass's [-0.03, 0.03] clamp override.

SIGN: our model returns E directly (TorchEBM convention), so we descend +grad_x E as
returned. (The UVA notebook computes -model(x) first only because ITS network returns
the score f = -E.)

OPTIONAL ANNEALING (noise_start -> noise_end across the chain):
Constant tiny noise makes Langevin behave like plain gradient descent: chains collapse
to one minimum (identical samples) and the final step leaves residual speckle. Annealing
fixes both -- high noise early = chains explore and spread out (diversity); low noise
late = chains settle cleanly (sharpness). anneal=False reproduces UVA exactly.
"""
import random

import numpy as np
import torch


class Sampler:
    """Persistent replay buffer + Langevin MCMC, following the UVA notebook."""

    def __init__(self, model, img_shape, sample_size, device, max_len=8192,
                 noise=0.005, new_sample_ratio=0.05):
        """
        Inputs:
            model            - TorchEBM BaseModel returning the ENERGY E
            img_shape        - Shape of the images to model, e.g. (1, 28, 28)
            sample_size      - Batch size of the samples
            device           - torch.device the samples/model live on
            max_len          - Maximum number of images to keep in the buffer
            noise            - per-step Langevin noise std (UVA: 0.005)
            new_sample_ratio - fraction of each batch started from FRESH noise (UVA: 0.05)
        """
        super().__init__()
        self.model = model
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.device = device
        self.max_len = max_len
        self.noise = noise
        self.new_sample_ratio = new_sample_ratio
        # The replay buffer: born as pure noise in [-1, 1], one image per sample slot.
        self.examples = [(torch.rand((1,) + img_shape) * 2 - 1) for _ in range(self.sample_size)]

    def sample_new_exmps(self, steps=60, step_size=10, anneal=False, noise_end=None):
        """A new batch of "fake" images (the CD negatives).

        95% resume from the replay buffer, 5% start from fresh noise -- so chains
        accumulate refinement across training steps (Persistent CD) instead of
        restarting cold every time. Refined by `steps` of Langevin, then written back.
        """
        n_new = np.random.binomial(self.sample_size, self.new_sample_ratio)
        rand_imgs = torch.rand((n_new,) + self.img_shape) * 2 - 1
        old_imgs = torch.cat(random.choices(self.examples, k=self.sample_size - n_new), dim=0)
        inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach().to(self.device)

        inp_imgs = Sampler.generate_samples(
            self.model, inp_imgs, steps=steps, step_size=step_size,
            noise=self.noise, anneal=anneal, noise_end=noise_end,
        )

        # Add the refined images to the FRONT of the buffer, trim to max_len.
        self.examples = list(inp_imgs.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.examples
        self.examples = self.examples[:self.max_len]
        return inp_imgs

    @staticmethod
    def generate_samples(model, inp_imgs, steps=60, step_size=10, noise=0.005,
                         anneal=False, noise_end=None, return_img_per_step=False):
        """Langevin MCMC: roll inp_imgs downhill on E (theta frozen), with noise.

        Args:
            model      - TorchEBM BaseModel returning ENERGY E (its .gradient() applies
                         the [-0.03, 0.03] clamp)
            steps      - number of Langevin steps
            step_size  - eta in  x <- x - eta * grad_x E
            noise      - per-step noise std (the STARTING std when anneal=True)
            anneal     - decay the noise std linearly from `noise` to `noise_end`
            noise_end  - final noise std when anneal=True (defaults to noise/50)
        """
        # Freeze params: we only move the INPUT here, never the weights (and it's cheaper).
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        inp_imgs = inp_imgs.clone().detach()
        # Reused noise buffer (re-filled each iteration) -- avoids reallocating.
        noise_buf = torch.randn(inp_imgs.shape, device=inp_imgs.device)
        if anneal and noise_end is None:
            noise_end = noise / 50.0

        imgs_per_step = []

        for i in range(steps):
            # Per-step noise std: constant (UVA) or annealed high -> low.
            if anneal and steps > 1:
                frac = i / (steps - 1)                       # 0 -> 1 across the chain
                std = noise + frac * (noise_end - noise)     # linear decay
            else:
                std = noise

            # Part 1: add exploration noise FIRST, keep pixels in [-1, 1].
            noise_buf.normal_(0, std)
            inp_imgs.add_(noise_buf)
            inp_imgs.clamp_(min=-1.0, max=1.0)

            # Part 2: grad_x E evaluated at the NOISED point. model.gradient() is
            # TorchEBM's autograd + our subclass's element-wise clamp.
            grad = model.gradient(inp_imgs)

            # Part 3: descent step  x <- x - step_size * grad_x E
            inp_imgs.add_(-step_size * grad)
            inp_imgs.clamp_(min=-1.0, max=1.0)

            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())

        # Restore params for training.
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)
        torch.set_grad_enabled(had_gradients_enabled)

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        return inp_imgs.detach()
