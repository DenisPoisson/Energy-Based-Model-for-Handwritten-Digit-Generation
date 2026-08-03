"""The energy network -- UVA's EXACT architecture, expressed as a TorchEBM BaseModel.

SIGN CONVENTION (differs from the UVA notebook on purpose -- read this):
    The UVA notebook's CNN outputs a SCORE f = -E (higher = more data-like), and its
    sampler negates internally. TorchEBM's convention is the opposite: a BaseModel's
    forward() returns the ENERGY E directly (LOWER = more data-like), and the library
    descends -grad_x E.

    We adopt TORCHEBM's convention here: forward() returns E.
    The architecture is bit-identical to UVA; only the sign bookkeeping moves. Doing it
    this way means every library component (gradient, sampler, sign_check) agrees, and
    inference NEVER needs a NEGATE flag.

Architecture notes (why it is shaped this way):
  - Channels come from ONE number: c_hid1 = hidden//2, c_hid2 = hidden, c_hid3 = hidden*2.
    With hidden=32 that is 16 -> 32 -> 64 -> 64 (the last two convs both use c_hid3).
  - The first conv is 5x5 stride 2 padding 4. The large padding GROWS the image before
    striding: (28 + 2*4 - 5)//2 + 1 = 16, so 28 -> 16 (not 14). A big kernel + big pad
    gives a wide receptive field in layer one -> broad stroke structure is seen early,
    which makes grad_x E (the force field Langevin follows) smoother.
  - Spatial funnel: 28 -> 16 -> 8 -> 4 -> 2, then Flatten -> FC -> act -> FC -> 1.
    The extra hidden FC layer is capacity in the head that a conv-only stack lacks.
"""
import torch
import torch.nn as nn

from torchebm.core import BaseModel


class Swish(nn.Module):
    """Smooth activation: x * sigmoid(x).  (Identical to nn.SiLU; kept explicit because
    it is what the UVA notebook uses by name.)

    Smoothness matters more here than in a classifier: Langevin moves images along
    grad_x E, so a kinked activation (ReLU) gives a piecewise-constant, jerky force
    field. Smooth activation -> smooth, navigable landscape.
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


def make_activation(name: str) -> nn.Module:
    """Config-driven activation. Default 'swish' reproduces UVA exactly."""
    acts = {
        "swish": Swish, "silu": nn.SiLU,      # same function, two names
        "relu": nn.ReLU, "gelu": nn.GELU, "softplus": nn.Softplus,
    }
    key = name.lower()
    if key not in acts:
        raise ValueError(f"Unknown activation {name!r}. Use one of {list(acts)}.")
    return acts[key]()


class CNNModel(BaseModel):
    """Image -> one scalar ENERGY E.  Exact UVA layer stack, config-driven width/act.

    Subclasses TorchEBM's BaseModel, so it inherits `.gradient(x)` (= grad_x E). We
    override `.gradient()` below to inject UVA's per-step gradient clamp.
    """

    def __init__(self, hidden_features: int = 32, out_dim: int = 1,
                 activation: str = "swish", grad_clamp: float | None = 0.03, **kwargs):
        super().__init__()
        self.grad_clamp = grad_clamp   # element-wise cap on grad_x E during sampling
        c_hid1 = hidden_features // 2
        c_hid2 = hidden_features
        c_hid3 = hidden_features * 2
        act = lambda: make_activation(activation)   # fresh module per layer

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, c_hid1, kernel_size=5, stride=2, padding=4),          # 28 -> 16
            act(),
            nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1),     # 16 -> 8
            act(),
            nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2, padding=1),     # 8 -> 4
            act(),
            nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2, padding=1),     # 4 -> 2
            act(),
            nn.Flatten(),
            nn.Linear(c_hid3 * 4, c_hid3),   # c_hid3 channels x 2 x 2 = c_hid3*4
            act(),
            nn.Linear(c_hid3, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 1, 28, 28) -> (B, 1) -> (B,) : one scalar ENERGY per image
        return self.cnn_layers(x).squeeze(dim=-1)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """grad_x E(x), with UVA's element-wise clamp injected.

        UVA clamps the input gradient to [-0.03, 0.03] every Langevin step, which caps
        how far any single pixel can be pushed -> the chain stays stable and the replay
        buffer never fills with garbage. Overriding .gradient() is the hook that applies
        it everywhere the model's input-gradient is used. Symmetric clamp = magnitude cap
        only, so the descent direction (sign) is unchanged.
        """
        g = super().gradient(x)
        if self.grad_clamp is not None:
            g = g.clamp(-self.grad_clamp, self.grad_clamp)
        return g
