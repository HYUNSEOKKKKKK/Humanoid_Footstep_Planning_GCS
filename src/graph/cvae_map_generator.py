"""
cvae_map_generator.py

Reproduction scaffold for the paper's CVAE-based map generator (Supplementary Materials).
Implements a Conditional VAE to learn the distribution of psi (6D stepping-stone relative pose)
conditioned on (psi_{n-2}, psi_{n-1}, T_last).

Key specs (from the paper):
- Encoder: MLP with hidden layers [512, 128]
- Decoder: MLP with hidden layers [128, 512]
- Loss: Total = MSE(psi, psi_hat) + 0.04 * KLD(q(z|psi,y) || N(0, I))
- Sampling difficulty control: alpha scales latent variance during sampling (z ~ N(0, alpha^2 I))

This file provides:
- CVAE model
- Simple synthetic dataset scaffold (replace with your logged psi sequences for real training)
- Training loop (PPO etc. is **not** implemented here; this module only handles ψ-generation)
- Sampling functions with alpha scaling
- Utilities to convert ψ sequences into 3D poses for stepping stones

Requirements:
    python >= 3.9
    torch >= 2.0
    numpy
    (optional) tqdm

Usage (example):
    python cvae_map_generator.py --epochs 50 --batch_size 512 --latent_dim 32 --alpha 0.8

Replace SyntheticPsiSequenceDataset with your real dataset by implementing
a Dataset that returns (psi, cond_y) tuples where:
    psi: shape [6] -> (r, theta, phi, delta_yaw, x_tilt, y_tilt)
    cond_y: shape [2*6 + 1] = [13] -> concat(psi_{n-2}, psi_{n-1}, T_last)

Coordinate conventions (summary from Supplementary text):
    - d_n: vector from center of stone_{n-1} to center of stone_n
    - x_n: ground-plane projection of d_n
    - Z_g: opposite to gravity (world +Z up)
    - (x_n^s, y_n^s, z_n^s): local axes of stone_n
    - r_n = ||x_n||
    - theta_n = angle(x_{n-1}, x_n)
    - phi_n = angle(d_n, x_n)
    - Final pose of stone_n: start aligned x_n^s // x_n, z_n^s // Z_g then sequential rotations:
        Δyaw_n around Z_g, x_tilt_n around x_n^s, y_tilt_n around y_n^s

NOTE: This is a *minimal* scaffold. For faithful reproduction you should:
    - feed real ψ sequences collected from successful tracker rollouts
    - normalize inputs per-dimension
    - tune optimizer, schedule, batch sizes
    - checkpoint & evaluate on held-out sets
"""

from dataclasses import dataclass
import argparse
import math
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm


# -------------------------------
# Utilities
# -------------------------------

def degrees_to_radians(arr):
    return np.deg2rad(arr)

def radians_to_degrees(arr):
    return np.rad2deg(arr)

def to_torch(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

# -------------------------------
# CVAE Model
# -------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, activation=nn.ReLU, final_activation=None):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), activation()]
            last = h
        layers.append(nn.Linear(last, out_dim))
        if final_activation is not None:
            layers.append(final_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class CVAE(nn.Module):
    def __init__(self, psi_dim=6, cond_dim=13, latent_dim=32, kld_weight=0.04):
        super().__init__()
        self.psi_dim = psi_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.kld_weight = kld_weight

        # Encoder: [psi ; cond] -> [512, 128] -> (mu, logvar)
        enc_in = psi_dim + cond_dim
        self.encoder = MLP(enc_in, [512, 128], 2 * latent_dim)

        # Decoder: [z ; cond] -> [128, 512] -> psi_hat
        dec_in = latent_dim + cond_dim
        self.decoder = MLP(dec_in, [128, 512], psi_dim)

    def encode(self, psi, cond):
        x = torch.cat([psi, cond], dim=-1)  # [B, psi+cond]
        enc_out = self.encoder(x)
        mu, logvar = torch.chunk(enc_out, 2, dim=-1)  # each [B, latent_dim]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        x = torch.cat([z, cond], dim=-1)
        psi_hat = self.decoder(x)
        return psi_hat

    def forward(self, psi, cond):
        mu, logvar = self.encode(psi, cond)
        z = self.reparameterize(mu, logvar)
        psi_hat = self.decode(z, cond)
        return psi_hat, mu, logvar

    def loss(self, psi, psi_hat, mu, logvar):
        # Reconstruction: MSE between input psi and decoded psi_hat
        recon = nn.functional.mse_loss(psi_hat, psi, reduction="mean")
        # KL divergence to unit normal
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon + self.kld_weight * kld
        return total, recon, kld

    @torch.no_grad()
    def sample(self, cond, alpha=1.0, device="cpu"):
        """
        Sample ψ ~ p(ψ | y) by drawing z ~ N(0, alpha^2 I).
        cond: [B, cond_dim]
        """
        B = cond.shape[0]
        z = torch.randn(B, self.latent_dim, device=device) * alpha
        psi_hat = self.decode(z, cond)
        return psi_hat

# -------------------------------
# Dataset (Synthetic scaffold)
# -------------------------------

@dataclass
class PsiRange:
    r_min: float = 0.4
    r_max: float = 1.6
    theta_min_deg: float = -60.0
    theta_max_deg: float = 60.0
    phi_min_deg: float = -60.0
    phi_max_deg: float = 60.0
    dyaw_min_deg: float = -45.0
    dyaw_max_deg: float = 45.0
    xtilt_min_deg: float = 0.0
    xtilt_max_deg: float = 90.0
    ytilt_min_deg: float = -30.0
    ytilt_max_deg: float = 30.0

class SyntheticPsiSequenceDataset(Dataset):
    """
    Produces synthetic (psi, cond_y) pairs to let you run the CVAE end-to-end.
    Replace this with your real dataset from tracker-success rollouts.
    """
    def __init__(self, length=20000, psi_range: PsiRange=PsiRange(), seed=0):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.length = length
        self.psi_range = psi_range

    def _sample_psi(self):
        pr = self.psi_range
        r = self.rng.uniform(pr.r_min, pr.r_max)
        theta = self.rng.uniform(pr.theta_min_deg, pr.theta_max_deg)
        phi = self.rng.uniform(pr.phi_min_deg, pr.phi_max_deg)
        dyaw = self.rng.uniform(pr.dyaw_min_deg, pr.dyaw_max_deg)
        xtilt = self.rng.uniform(pr.xtilt_min_deg, pr.xtilt_max_deg)
        ytilt = self.rng.uniform(pr.ytilt_min_deg, pr.ytilt_max_deg)
        # store radians for angles
        psi = np.array([
            r,
            math.radians(theta),
            math.radians(phi),
            math.radians(dyaw),
            math.radians(xtilt),
            math.radians(ytilt),
        ], dtype=np.float32)
        return psi

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Randomly create previous two ψ and current ψ
        psi_nm2 = self._sample_psi()
        psi_nm1 = self._sample_psi()
        psi_n = self._sample_psi()

        # T_last: 10% chance that this is last target (1.0), else 0.0
        T_last = float(self.rng.random() < 0.1)

        # Condition vector y = [psi_{n-2}, psi_{n-1}, T_last]
        cond = np.concatenate([psi_nm2, psi_nm1, np.array([T_last], dtype=np.float32)], axis=0)

        return (
            torch.tensor(psi_n, dtype=torch.float32),
            torch.tensor(cond, dtype=torch.float32)
        )

# -------------------------------
# Pose reconstruction from ψ sequence
# -------------------------------

def rotation_matrix(axis, angle):
    """
    Rodrigues' rotation formula for unit axis.
    """
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=np.float64)
    I = np.eye(3)
    R = I + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
    return R

def build_poses_from_psi_sequence(psi_seq, x_prev=np.array([1.0, 0.0, 0.0]), p0=np.zeros(3)):
    """
    Given a sequence of ψ (each [r, theta, phi, dyaw, xtilt, ytilt] in radians, distances in meters),
    reconstruct stepping-stone centers and axes.

    Returns:
        centers: [N, 3]
        axes: [N, 3, 3]  (columns are x_s, y_s, z_s unit vectors for each stone)
    """
    Zg = np.array([0.0, 0.0, 1.0])  # opposite to gravity (up)
    centers = []
    axes_all = []

    x_prev = x_prev / np.linalg.norm(x_prev)
    p = p0.copy()

    for psi in psi_seq:
        r, theta, phi, dyaw, xtilt, ytilt = psi  # angles already in radians

        # x_n: rotate previous ground-direction by theta
        c, s = math.cos(theta), math.sin(theta)
        rot_z = np.array([[c, -s, 0],
                          [s,  c, 0],
                          [0,  0, 1]], dtype=np.float64)
        x_n = rot_z @ x_prev
        x_n = x_n / np.linalg.norm(x_n)

        # d_n: vector with ground magnitude r and elevation per phi
        # If phi is angle between d_n and x_n, then |d_n| = r / cos(phi)
        # Horizontal component magnitude = r, vertical component = |d_n| * sin(phi)
        d_mag = r / max(math.cos(phi), 1e-6)
        d_n = d_mag * np.array([x_n[0], x_n[1], 0.0]) + (d_mag * math.sin(phi)) * Zg

        p = p + d_n  # new center
        centers.append(p.copy())

        # Initial local axes x_s // x_n, z_s // Zg
        x_s = x_n / np.linalg.norm(x_n)
        z_s = Zg / np.linalg.norm(Zg)
        y_s = np.cross(z_s, x_s)
        y_s /= np.linalg.norm(y_s)

        # Sequential rotations: Δyaw around Z_g, then x_tilt around x_s, then y_tilt around y_s
        R_yaw = rotation_matrix(Zg, dyaw)
        x_s = R_yaw @ x_s
        y_s = R_yaw @ y_s
        z_s = R_yaw @ z_s

        R_xtilt = rotation_matrix(x_s, xtilt)
        y_s = R_xtilt @ y_s
        z_s = R_xtilt @ z_s  # x_s stays same

        R_ytilt = rotation_matrix(y_s, ytilt)
        x_s = R_ytilt @ x_s
        z_s = R_ytilt @ z_s  # y_s stays same

        # Orthonormalize (numerical safety)
        x_s = x_s / np.linalg.norm(x_s)
        z_s = z_s / np.linalg.norm(z_s)
        y_s = np.cross(z_s, x_s)
        y_s = y_s / np.linalg.norm(y_s)

        axes_all.append(np.stack([x_s, y_s, z_s], axis=1))

        # Prepare next iteration
        x_prev = x_n

    return np.stack(centers, axis=0), np.stack(axes_all, axis=0)

# -------------------------------
# Training loop
# -------------------------------

def train_cvae(model, loader, device, epochs=50, lr=1e-3, grad_clip=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(1, epochs + 1):
        total, recon_total, kld_total = 0.0, 0.0, 0.0
        for psi, cond in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
            psi = psi.to(device)
            cond = cond.to(device)
            psi_hat, mu, logvar = model(psi, cond)
            loss, recon, kld = model.loss(psi, psi_hat, mu, logvar)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            total += loss.item() * psi.size(0)
            recon_total += recon.item() * psi.size(0)
            kld_total += kld.item() * psi.size(0)

        n = len(loader.dataset)
        print(f"[{epoch:03d}] loss={total/n:.6f}  recon={recon_total/n:.6f}  kld={kld_total/n:.6f}")

# -------------------------------
# Sampling demo
# -------------------------------

@torch.no_grad()
def demo_sampling(model, device, alpha=1.0, num=5):
    model.eval()
    # Dummy condition y (two previous psis + T_last)
    psi_dim, cond_dim = model.psi_dim, model.cond_dim
    # Build random cond for demo (replace with real sequence context in practice)
    cond = torch.zeros(num, cond_dim, device=device)
    # Randomize a bit so samples differ
    cond.normal_(mean=0.0, std=0.1)
    psi_samples = model.sample(cond, alpha=alpha, device=device).cpu().numpy()
    return psi_samples

# -------------------------------
# CLI
# -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1.0, help="variance scaling for sampling")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_size", type=int, default=20000)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    psi_dim = 6
    cond_dim = 2 * psi_dim + 1  # (psi_{n-2}, psi_{n-1}, T_last)

    model = CVAE(psi_dim=psi_dim, cond_dim=cond_dim, latent_dim=args.latent_dim, kld_weight=0.04)
    model.to(args.device)

    # Synthetic dataset (replace with your real data loader)
    dataset = SyntheticPsiSequenceDataset(length=args.train_size, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True)

    print("Training CVAE (synthetic data) ...")
    train_cvae(model, loader, device=args.device, epochs=args.epochs, lr=args.lr, grad_clip=1.0)

    print("\nSampling demo with alpha =", args.alpha)
    samples = demo_sampling(model, device=args.device, alpha=args.alpha, num=8)
    print("Sampled psi (r, theta, phi, dyaw, xtilt, ytilt) in radians:")
    np.set_printoptions(precision=4, suppress=True)
    print(samples)

    # Build example poses from a small psi sequence
    psi_seq = samples[:4]  # 4 stones for demo
    centers, axes = build_poses_from_psi_sequence(psi_seq)
    print("\nExample 3D centers from sampled ψ sequence:")
    print(centers)
    print("\nExample local axes (x_s, y_s, z_s columns) for each stone:")
    print(axes)

if __name__ == "__main__":
    main()
