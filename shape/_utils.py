import torch

from core import fmt_param, term


def fmt_curv_pair(c: torch.Tensor) -> str:
    c = c.detach().cpu()
    r = torch.where(c.abs().gt(1e-12), c.reciprocal(), torch.tensor(float("inf")))
    return (
        f"{term.CURVATURE.canonical}={fmt_param(c)},\n"
        f"{term.RADIUS.canonical}={fmt_param(r)}"
    )


def fmt_alpha_phys(alpha: torch.Tensor, radius: torch.Tensor) -> str:
    alpha = alpha.detach().cpu()
    radius = radius.detach().cpu().unsqueeze(-1)
    n_coeffs = alpha.shape[-1]
    power = torch.arange(
        4, 4 + n_coeffs * 2, step=2, device=alpha.device, dtype=alpha.dtype
    )
    phys = alpha * radius.pow(power)

    lines: list[str] = []
    for i in range(n_coeffs):
        order = 4 + i * 2
        tag = f"{term.ALPHA.canonical}[{order}]"
        lines.append(f"{tag}_norm={fmt_param(alpha[:, i])}")
        lines.append(f"{tag}_phys={fmt_param(phys[:, i])}")
    return ",\n".join(lines)
