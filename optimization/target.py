from dataclasses import dataclass
from typing import Any, Self

from core import term

type FOV = float | tuple[float, float] | tuple[tuple[float, float], tuple[float, float]]


def _parse_fov(raw: Any) -> FOV:
    match raw:
        case int() | float():
            return float(raw)
        case [a, b] | (a, b) if isinstance(a, (int, float)):
            return (float(a), float(b))
        case [[a, b], [c, d]] | ((a, b), (c, d)):
            return ((float(a), float(b)), (float(c), float(d)))
        case _:
            raise TypeError(f"Invalid fov: {raw!r}")


def _serialize_fov(fov: FOV) -> float | list[float] | list[list[float]]:
    match fov:
        case float():
            return fov
        case (float() as a, float() as b):
            return [a, b]
        case ((float() as a, float() as b), (float() as c, float() as d)):
            return [[a, b], [c, d]]
        case _:
            raise RuntimeError(f"unreachable: {fov!r}")


def _fov_name(fov: FOV) -> str:
    match fov:
        case float() as f:
            return f"{f:g}"
        case (float() as a, float() as b):
            return f"{a:g}_{b:g}"
        case (
            (float() as h_min, float() as h_max),
            (float() as v_min, float() as v_max),
        ):
            return f"{h_min:g}_{h_max:g}_{v_min:g}_{v_max:g}"
        case _:
            raise RuntimeError(f"unreachable: {fov!r}")


@dataclass(slots=True)
class Target:
    id: str
    fov: FOV
    F: float
    effl: float
    wavelengths: list[float]  # nm

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        fov = _parse_fov(term.FOV.resolve(data))
        F = float(term.F_NUMBER.resolve(data))
        effl = float(term.EFFL.resolve(data))

        wavelengths = [float(w) for w in term.WAVELENGTH.resolve(data, default=[550.0])]

        id = term.ID.resolve(
            data,
            default=f"fov_{_fov_name(fov)}_F_{F:g}_effl_{effl:g}",
        )

        return cls(
            id=id,
            fov=fov,
            F=F,
            effl=effl,
            wavelengths=wavelengths,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            str(term.ID.canonical): self.id,
            str(term.FOV.canonical): _serialize_fov(self.fov),
            str(term.F_NUMBER.canonical): self.F,
            str(term.EFFL.canonical): self.effl,
            str(term.EPD.canonical): self.epd,
            str(term.WAVELENGTH.canonical): self.wavelengths,
        }
        return d

    @property
    def epd(self) -> float:
        return self.effl / self.F

    def __repr__(self) -> str:
        return (
            f"Target({term.ID}: {self.id!r}, "
            f"{term.FOV}: {self.fov!r}, "
            f"{term.F_NUMBER}: {self.F:g}, "
            f"{term.EFFL}: {self.effl:g}, "
            f"{term.EPD}: {self.effl:g}, "
            f"{term.WAVELENGTH}: {self.wavelengths!r})"
        )
