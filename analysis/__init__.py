"""像质分析模块：PSF / 派生指标（全程无梯度，float64）。

模块组织
--------
* :mod:`analysis._kernel` —— Kirchhoff 波前内核（追迹 + 参考球 + 积分器）。
* :mod:`analysis.psf` —— 单色 / 多色 PSF 主管线（直评，无重采样）：
  :class:`PsfSpec` 规格 + :func:`psf` / :func:`psf_from_checkpoint` 入口。
  设计文档见 ``docs/psf-redesign.md``。
* :mod:`analysis.metrics` —— MTF / 波前误差 / Strehl 比。
"""

from analysis.metrics import MtfResult, WavefrontResult, mtf, strehl, wavefront
from analysis.psf import Issue, PsfReport, PsfSpec, psf, psf_from_checkpoint

__all__ = [
    "Issue",
    "MtfResult",
    "PsfReport",
    "PsfSpec",
    "WavefrontResult",
    "mtf",
    "psf",
    "psf_from_checkpoint",
    "strehl",
    "wavefront",
]
