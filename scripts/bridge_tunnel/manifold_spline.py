"""Minimal 1D spline-manifold, ported from goodfire-ai/causalab (manifold_steering
branch): causalab/methods/spline/{cubic,manifold}.py and
causalab/methods/pullback/geodesic.py.

Trimmed to exactly the case this project needs: a 1D natural cubic spline
(Reinsch 1967 smoothing spline) through binned centroids in a plain ambient
space (no sphere projection, no periodic BC, no TPS), plus Gauss-Newton
projection onto the manifold and the manifold-trace ("geodesic in intrinsic
coordinate") path used for manifold steering. No pyvene / no causalab I/O
dependency; pure torch so it runs against the Dreamer belief feats directly.

Attribution: algorithm and structure are from causalab (Goodfire); this file
is a standalone re-implementation of just the 1D/plain path.
"""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn


class CubicSpline1D(nn.Module):
    """Natural cubic smoothing spline (Reinsch). control_points: (n,1) knots;
    values: (n, ambient). smoothness=0 -> exact interpolation."""

    def __init__(self, control_points: Tensor, values: Tensor, smoothness: float = 0.0):
        super().__init__()
        assert control_points.ndim == 2 and control_points.shape[1] == 1
        n = control_points.shape[0]
        assert n >= 2 and values.shape[0] == n
        self.smoothness = float(smoothness)
        self.ambient_dim = values.shape[1]

        x_raw = control_points[:, 0]
        sort_idx = torch.argsort(x_raw)
        x_sorted = x_raw[sort_idx].contiguous()
        y_sorted = values[sort_idx].contiguous()
        h = x_sorted[1:] - x_sorted[:-1]
        if torch.any(h <= 0):
            raise ValueError("control_points must be strictly increasing (no dup knots)")

        self.register_buffer("x_sorted", x_sorted)
        self.register_buffer("h", h)
        gamma, y_hat = self._fit_natural(x_sorted, y_sorted, h, self.smoothness)
        self.register_buffer("gamma", gamma)
        self.register_buffer("y_hat", y_hat)

    @staticmethod
    def _fit_natural(x, y, h, lam):
        n = x.shape[0]
        device, dtype = y.device, y.dtype
        ambient = y.shape[1]
        if n == 2:
            return torch.zeros(n, ambient, device=device, dtype=dtype), y.clone()
        m = n - 2
        diag_R = (h[:-1] + h[1:]) / 3.0
        off_R = h[1:-1] / 6.0
        R = torch.diag(diag_R) + torch.diag(off_R, 1) + torch.diag(off_R, -1)
        inv_h = 1.0 / h
        Qt_y = (
            y[:-2] * inv_h[:-1].unsqueeze(-1)
            - y[1:-1] * (inv_h[:-1] + inv_h[1:]).unsqueeze(-1)
            + y[2:] * inv_h[1:].unsqueeze(-1)
        )
        if lam == 0.0:
            A = R
        else:
            Q = torch.zeros(n, m, device=device, dtype=dtype)
            for j in range(m):
                Q[j, j] = inv_h[j]
                Q[j + 1, j] = -(inv_h[j] + inv_h[j + 1])
                Q[j + 2, j] = inv_h[j + 1]
            A = R + lam * (Q.T @ Q)
        gamma_int = torch.linalg.solve(A, Qt_y)
        gamma = torch.zeros(n, ambient, device=device, dtype=dtype)
        gamma[1:-1] = gamma_int
        if lam == 0.0:
            y_hat = y.clone()
        else:
            Qg = torch.zeros_like(y)
            for j in range(m):
                Qg[j] += inv_h[j] * gamma_int[j]
                Qg[j + 1] += -(inv_h[j] + inv_h[j + 1]) * gamma_int[j]
                Qg[j + 2] += inv_h[j + 1] * gamma_int[j]
            y_hat = y - lam * Qg
        return gamma, y_hat

    @staticmethod
    def _eval_segment(u, x_left, x_right, y_l, y_r, g_l, g_r):
        h_seg = (x_right - x_left).unsqueeze(-1)
        a = (x_right - u).unsqueeze(-1)
        b = (u - x_left).unsqueeze(-1)
        return (
            g_l * (a ** 3) / (6.0 * h_seg)
            + g_r * (b ** 3) / (6.0 * h_seg)
            + (y_l / h_seg - g_l * h_seg / 6.0) * a
            + (y_r / h_seg - g_r * h_seg / 6.0) * b
        )

    def evaluate(self, u: Tensor) -> Tensor:
        if u.ndim != 2 or u.shape[1] != 1:
            raise ValueError(f"expects (batch,1), got {tuple(u.shape)}")
        u_flat = u[:, 0]
        x, h, gamma, y_hat = self.x_sorted, self.h, self.gamma, self.y_hat
        n = x.shape[0]
        u_clamped = u_flat.clamp(x[0], x[-1])
        idx = torch.searchsorted(x, u_clamped, right=True) - 1
        idx = idx.clamp(0, n - 2)
        out = self._eval_segment(u_clamped, x[idx], x[idx + 1],
                                 y_hat[idx], y_hat[idx + 1], gamma[idx], gamma[idx + 1])
        # linear extrapolation outside knot range
        h0, hn = h[0], h[-1]
        slope_left = (y_hat[1] - y_hat[0]) / h0 - h0 * gamma[1] / 6.0
        slope_right = (y_hat[-1] - y_hat[-2]) / hn + hn * gamma[-2] / 6.0
        delta_left = (u_flat - x[0]).clamp(max=0.0).unsqueeze(-1)
        delta_right = (u_flat - x[-1]).clamp(min=0.0).unsqueeze(-1)
        return out + slope_left.unsqueeze(0) * delta_left + slope_right.unsqueeze(0) * delta_right


class SplineManifold1D(nn.Module):
    """1D manifold: intrinsic coord u (== the conceptual coordinate z) -> ambient
    belief feat. control_z: (n,) knot positions; centroids: (n, ambient) bin
    means. Provides decode, Gauss-Newton projection, and manifold-trace path."""

    def __init__(self, control_z: Tensor, centroids: Tensor, smoothness: float = 0.0):
        super().__init__()
        cp = control_z.reshape(-1, 1).float()
        self.register_buffer("control_z", cp)
        self.register_buffer("centroids", centroids.float())
        self.spline = CubicSpline1D(cp, centroids.float(), smoothness=smoothness)
        self._ambient = centroids.shape[1]

    def decode(self, u: Tensor) -> Tensor:
        if u.ndim == 1:
            u = u.unsqueeze(-1)
        return self.spline.evaluate(u.to(self.control_z.device))

    def encode_nearest_centroid(self, x: Tensor):
        d = torch.cdist(x, self.centroids, p=2)
        idx = d.argmin(dim=1)
        return self.control_z[idx].clone(), d[torch.arange(x.shape[0]), idx]

    def encode_to_nearest_point(self, x: Tensor, n_iters: int = 8, tol: float = 1e-6,
                                damping: float = 1e-6):
        """Gauss-Newton: argmin_u ||decode(u)-x||^2 from nearest-centroid warm start.
        Returns (u:(batch,1), residual_vec:(batch,ambient))."""
        x = x.to(self.control_z.device)
        u, _ = self.encode_nearest_centroid(x)  # (batch,1)
        u = u.clone().detach()
        fd = 1e-4
        for _ in range(n_iters):
            r = self.decode(u) - x                     # (batch, ambient)
            u_f = u.clone(); u_f[:, 0] += fd
            u_b = u.clone(); u_b[:, 0] -= fd
            J = (self.decode(u_f) - self.decode(u_b)) / (2 * fd)   # (batch, ambient)
            JtJ = (J * J).sum(-1, keepdim=True) + damping           # (batch,1)
            Jtr = (J * r).sum(-1, keepdim=True)                     # (batch,1)
            delta = Jtr / JtJ
            u = u - delta
            if delta.abs().max() < tol:
                break
        residual = x - self.decode(u)
        return u, residual

    def off_manifold_distance(self, x: Tensor) -> Tensor:
        _, res = self.encode_to_nearest_point(x)
        return res.norm(dim=-1)

    def manifold_trace_path(self, z_a: float, z_b: float, n_steps: int) -> Tensor:
        """Path ON the manifold from the projection of z_a's decoded point to
        z_b's: interpolate the intrinsic coordinate u linearly, decode. (For a
        1D manifold the projected endpoints are essentially u=z_a, u=z_b.)"""
        dev = self.control_z.device
        x_a = self.decode(torch.tensor([[float(z_a)]], device=dev))
        x_b = self.decode(torch.tensor([[float(z_b)]], device=dev))
        u_a, _ = self.encode_to_nearest_point(x_a)
        u_b, _ = self.encode_to_nearest_point(x_b)
        t = torch.linspace(0, 1, n_steps, device=dev).unsqueeze(1)
        u_path = u_a + t * (u_b - u_a)          # (n_steps,1)
        return self.decode(u_path)               # (n_steps, ambient)

    def linear_path(self, z_a: float, z_b: float, n_steps: int) -> Tensor:
        """Straight line in AMBIENT space between the two endpoints' decoded
        belief feats -- the linear-steering baseline."""
        dev = self.control_z.device
        x_a = self.decode(torch.tensor([[float(z_a)]], device=dev))[0]
        x_b = self.decode(torch.tensor([[float(z_b)]], device=dev))[0]
        t = torch.linspace(0, 1, n_steps, device=dev).unsqueeze(1)
        return x_a.unsqueeze(0) + t * (x_b - x_a).unsqueeze(0)
