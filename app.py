import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from typing import Dict, Tuple

# ============================================================
# 0. CONFIGURATION GLOBALE (CONSTANTES & INVARIANTS)
# ============================================================

class MolecularConfig:
    def __init__(
        self,
        eps_forward: float = 1e-8,
        eps_tv: float = 1e-8,
        eps_fisher: float = 1e-5,
        gaussian_switch_sigma: float = 1e-4,
    ):
        self.eps_forward = float(eps_forward)
        self.eps_tv = float(eps_tv)
        self.eps_fisher = float(eps_fisher)
        self.gaussian_switch_sigma = float(gaussian_switch_sigma)


# ============================================================
# 1. OPÉRATEUR MOLÉCULAIRE ULTRA-COMPILÉ (ZÉRO BRANCHE)
# ============================================================

class MolecularOperatorFourier(nn.Module):
    """Opérateur de convolution basé sur la RFFT 2D, périodique, fullgraph-friendly."""

    def __init__(
        self,
        psf: torch.Tensor,
        img_shape: Tuple[int, int],
        attenuation: float = 0.0,
        config: MolecularConfig | None = None,
    ):
        super().__init__()
        self.H, self.W = img_shape
        self.config = config or MolecularConfig()

        if psf.ndim != 2:
            raise ValueError("La PSF doit être 2D (H_psf, W_psf).")

        psf = psf.to(dtype=torch.float32)
        psf_sum = psf.sum()
        if psf_sum <= 1e-12:
            raise ValueError("La somme de la PSF doit être positive.")
        psf_norm = psf / psf_sum

        ph, pw = psf_norm.shape
        pad_h, pad_w = self.H - ph, self.W - pw
        if pad_h < 0 or pad_w < 0:
            raise ValueError("La taille de la PSF ne peut pas dépasser la taille de l'image.")

        psf_padded = F.pad(psf_norm, (0, pad_w, 0, pad_h))
        psf_padded = torch.roll(psf_padded, shifts=(-(ph // 2), -(pw // 2)), dims=(-2, -1))

        H_f = torch.fft.rfft2(psf_padded, dim=(-2, -1))
        self.register_buffer("H_f", H_f, persistent=True)
        self.register_buffer("H_f_conj", torch.conj(H_f), persistent=True)

        attr_val = float(torch.exp(torch.tensor(-float(attenuation)))) if attenuation > 0.0 else 1.0
        self.register_buffer("attr", torch.tensor(attr_val, dtype=torch.float32), persistent=True)

        self.H_int = int(self.H)
        self.W_int = int(self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        X_f = torch.fft.rfft2(x, dim=(-2, -1))
        y = torch.fft.irfft2(X_f * self.H_f, s=(self.H_int, self.W_int), dim=(-2, -1))
        return y * self.attr

    def adjoint(self, x: torch.Tensor) -> torch.Tensor:
        X_f = torch.fft.rfft2(x, dim=(-2, -1))
        y = torch.fft.irfft2(X_f * self.H_f_conj, s=(self.H_int, self.W_int), dim=(-2, -1))
        return y * self.attr


# ============================================================
# 2. MODÈLE STATISTIQUE DE BRUIT
# ============================================================

class MolecularForwardModel(nn.Module):
    def __init__(
        self,
        operator: MolecularOperatorFourier,
        poisson_scale: float = 1.0,
        gaussian_sigma: float = 1e-3,
        config: MolecularConfig | None = None,
    ):
        super().__init__()
        self.op = operator
        self.poisson_scale = float(poisson_scale)
        self.gaussian_sigma = float(gaussian_sigma)
        self.config = config or MolecularConfig()
        self.eps = self.config.eps_forward

    def simulate(self, c: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            Ac = torch.clamp(self.op(c), min=self.eps)
            y = torch.poisson(Ac * self.poisson_scale) / self.poisson_scale
            if self.gaussian_sigma > 0.0:
                y = y + torch.randn_like(y) * self.gaussian_sigma
            return torch.clamp(y, min=0.0)

    def nll(self, c: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        Ac = torch.clamp(self.op(c), min=self.eps)
        sigma2 = self.gaussian_sigma ** 2

        nll_poisson = torch.sum(Ac - y * torch.log(Ac), dim=(-2, -1))

        var = torch.clamp(Ac + sigma2, min=self.eps)
        nll_gauss = 0.5 * torch.sum(((Ac - y) ** 2) / var + torch.log(var), dim=(-2, -1))

        switch = float(self.gaussian_sigma > self.config.gaussian_switch_sigma)
        switch_t = torch.tensor(switch, dtype=Ac.dtype, device=Ac.device)
        return (1.0 - switch_t) * nll_poisson + switch_t * nll_gauss


# ============================================================
# 3. SOLVEUR PROXIMAL ULTRA-COMPILÉ MONOLITHIQUE
# ============================================================

@torch.compile(mode="max-autotune", fullgraph=True)
def _fista_loop_compiled(
    y: torch.Tensor,
    c: torch.Tensor,
    z: torch.Tensor,
    n_steps: int,
    lr: float,
    lambda_tv: float,
    sigma2: float,
    eps_forward: float,
    eps_tv: float,
    switch_t: torch.Tensor,
    H_f: torch.Tensor,
    H_f_conj: torch.Tensor,
    attr: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    
    t = torch.ones((), dtype=y.dtype, device=y.device)
    
    for _ in range(n_steps):
        # 1. Opérateur Forward
        X_f = torch.fft.rfft2(z, dim=(-2, -1))
        Ac = torch.fft.irfft2(X_f * H_f, s=(H, W), dim=(-2, -1)) * attr
        Ac_clamped = torch.clamp(Ac, min=eps_forward)

        # 2. Résidu mixte sans branchement
        residual_poisson = 1.0 - y / Ac_clamped
        var = torch.clamp(Ac_clamped + sigma2, min=eps_forward)
        diff = Ac_clamped - y
        inv_var = 1.0 / var
        residual_gauss = diff * inv_var * (1.0 - 0.5 * diff * inv_var) + 0.5 * inv_var
        residual = (1.0 - switch_t) * residual_poisson + switch_t * residual_gauss

        # 3. Opérateur Adjoint
        R_f = torch.fft.rfft2(residual, dim=(-2, -1))
        grad_data = torch.fft.irfft2(R_f * H_f_conj, s=(H, W), dim=(-2, -1)) * attr

        # 4. Variation Totale périodique optimisée
        dx = torch.roll(z, shifts=-1, dims=-1) - z
        dy = torch.roll(z, shifts=-1, dims=-2) - z

        norm = torch.sqrt(dx * dx + dy * dy + eps_tv)
        nx = dx / norm
        ny = dy / norm

        div_x = nx - torch.roll(nx, shifts=1, dims=-1)
        div_y = ny - torch.roll(ny, shifts=1, dims=-2)
        grad_tv = -(div_x + div_y)

        # 5. Descente Proximale et accélération FISTA unifiée
        c_next = torch.clamp(z - lr * (grad_data + lambda_tv * grad_tv), min=0.0)
        t_next = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * t * t))
        weight = (t - 1.0) / t_next

        z = torch.clamp(c_next + weight * (c_next - c), min=0.0)
        c = c_next
        t = t_next
        
    return c


class AcceleratedMolecularReconstructor(nn.Module):
    def __init__(
        self,
        fm: MolecularForwardModel,
        lambda_tv: float = 1e-4,
        lr: float = 2e-1,
    ):
        super().__init__()
        self.fm = fm
        self.lambda_tv = float(lambda_tv)
        self.lr = float(lr)

    def forward(self, y: torch.Tensor, n_steps: int = 80) -> torch.Tensor:
        c = torch.clamp(y, min=0.01)
        z = c.clone()
        
        sigma2 = self.fm.gaussian_sigma ** 2
        switch_bool = float(self.fm.gaussian_sigma > self.fm.config.gaussian_switch_sigma)
        switch_t = torch.tensor(switch_bool, dtype=y.dtype, device=y.device)

        c_opt = _fista_loop_compiled(
            y, c, z, n_steps, self.lr, self.lambda_tv, sigma2,
            self.fm.eps, self.fm.config.eps_tv, switch_t, 
            self.fm.op.H_f, self.fm.op.H_f_conj, self.fm.op.attr, 
            self.fm.op.H_int, self.fm.op.W_int
        )

        return c_opt


# ============================================================
# 4. QUANTIFICATION D'INCERTITUDE ET MOTEUR UNIFIÉ
# ============================================================

class MolecularUncertainty(nn.Module):
    def __init__(self, fm: MolecularForwardModel):
        super().__init__()
        self.fm = fm

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            Ac = torch.clamp(self.fm.op(c), min=self.fm.eps)
            fisher_obs = self.fm.poisson_scale / (Ac + (self.fm.gaussian_sigma ** 2))
            fisher_map = self.fm.op.adjoint(fisher_obs)
            return 1.0 / torch.sqrt(torch.clamp(fisher_map, min=self.fm.config.eps_fisher))


class MolecularImagingEngine(nn.Module):
    def __init__(self, fm: MolecularForwardModel, lr: float = 2e-1, lambda_tv: float = 1e-4):
        super().__init__()
        self.fm = fm
        self.rec = AcceleratedMolecularReconstructor(fm, lambda_tv=lambda_tv, lr=lr)
        self.unc = MolecularUncertainty(fm)

    def reconstruct(self, y: torch.Tensor, n_steps: int = 80) -> Dict[str, torch.Tensor]:
        with torch.inference_mode():  # Utilisation exclusive de inference_mode à la place de no_grad
            c_hat = self.rec(y, n_steps=n_steps)
            crb_map = self.unc(c_hat)
        return {
            "concentration_map": c_hat,
            "uncertainty_map": crb_map,
        }


# ============================================================
# 5. EXECUTION DE VALIDATION
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    H, W = 128, 128
    config = MolecularConfig()

    x_coord = torch.linspace(-2, 2, 9, device=device, dtype=dtype)
    indexing_mode = "ij"
    xx, yy = torch.meshgrid(x_coord, x_coord, indexing=indexing_mode)
    psf_tensor = torch.exp(-(xx**2 + yy**2) / 2.0)

    op_fourier = MolecularOperatorFourier(psf_tensor, (H, W), attenuation=0.05, config=config).to(device)
    fm = MolecularForwardModel(op_fourier, poisson_scale=500.0, gaussian_sigma=0.001, config=config).to(device)
    engine = MolecularImagingEngine(fm, lr=1.5, lambda_tv=1e-4).to(device)

    c_true = torch.zeros((1, 1, H, W), device=device, dtype=dtype)
    c_true[:, :, 35:65, 35:65] = 2.0
    c_true[:, :, 80:105, 20:45] = 1.2

    y = fm.simulate(c_true)

    print("Warmup & Compilation globale sans rupture de graphe...")
    _ = engine.reconstruct(y, n_steps=5)

    print("Exécution de la reconstruction de précision (Fusion Totale)...")
    results = engine.reconstruct(y, n_steps=60)

    c_rec = results["concentration_map"]
    unc_map = results["uncertainty_map"]

    print("\n=== MOLECULAR IMAGING ENGINE v12.1 (Pure Monolithic Loop Compilation) ===")
    print(f"Cible d'exécution           : {device}")
    print(f"Min/Max Concentration       : {c_rec.min().item():.4f} / {c_rec.max().item():.4f}")
    print(f"Incertitude Moyenne CRB    : {unc_map.mean().item():.6f}")
    print("=========================================================\n")
