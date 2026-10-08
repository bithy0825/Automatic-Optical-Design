"""像质分析模块：PSF / 相机合成 / 派生指标（全程无梯度，float64）。

模块组织
--------
* :mod:`analysis.grid` —— :class:`GridSpec`：显式像面采样网格。
* :mod:`analysis._kernel` —— Kirchhoff 波前内核（追迹 + 参考球 + 积分器）。
* :mod:`analysis.psf` —— 单色 / 多色 PSF 主管线（直评，无重采样）。
* :mod:`analysis.camera` —— 光谱响应与 RGB 相机合成。
* :mod:`analysis.metrics` —— MTF / 波前误差 / Strehl 比。
"""

from analysis.camera import CameraPsf, SensorResponse, camera_psf
from analysis.grid import GridSpec
from analysis.metrics import MtfResult, WavefrontResult, mtf, strehl, wavefront
from analysis.psf import Issue, PsfResult, psf

__all__ = [
    "CameraPsf",
    "GridSpec",
    "Issue",
    "MtfResult",
    "PsfResult",
    "SensorResponse",
    "WavefrontResult",
    "camera_psf",
    "mtf",
    "psf",
    "strehl",
    "wavefront",
]
