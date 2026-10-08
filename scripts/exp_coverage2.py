"""覆盖率常数标定实验：近衍射极限（小 EPD）下对照 Airy 解析包围能量。"""

import torch

from analysis import PsfSpec, psf
from analysis._kernel import eval_chain, reference_sphere, trace_with_opl
from component import Gap, InfiniteSource, Refractor, Sensor, Sequential
from materials import Material
from shape import Disk, Sphere


def build(epd: float, gap_img: float) -> Sequential:
    seq = Sequential(
        InfiniteSource(epd=epd, field_x=0.0, field_y=0.0, wavelength=555.0, population=1),
        Refractor(
            shape=Sphere(diameter=torch.full((1,), 10.0), curvature=torch.full((1,), 0.02)),
            transmitted=Material.from_name(1, "N-BK7"),
        ),
        Gap(5.0),
        Refractor(
            shape=Disk(diameter=torch.full((1,), 10.0)),
            transmitted=Material.from_name(1, "air"),
        ),
        Gap(gap_img),
        Sensor(shape=Disk(diameter=torch.full((1,), 5.0))),
    )
    seq.rebind()
    return seq


def main() -> None:
    # BFL ≈ R/(n−1) − t/n ≈ 96.3 − 3.3 ≈ 93.0
    for epd, gap_img, tag in ((1.0, 93.0, "近衍射极限"), (9.6, 93.0, "全口径")):
        seq = build(epd, gap_img)
        rep = psf(seq, PsfSpec(size=64))
        na = rep.na.flatten()[0].item()
        airy = 0.61 * 555e-6 / na  # Airy 半径 mm
        half = 0.5 * rep.size * rep.delta
        print(
            f"{tag}: epd={epd} NA={na:.5f} delta={rep.delta:.6f} "
            f"窗口半宽={half:.4f}mm = {half / airy:.1f}×Airy  coverage={rep.coverage.flatten()[0].item():.4f} "
            f"throughput={rep.throughput.flatten()[0].item():.4f}"
        )
        # 参考球半径 vs 实际末面-传感器距离
        ev = eval_chain(seq, None, (555.0,), (0.0, 0.0))
        tr = trace_with_opl(ev, None)
        sp = reference_sphere(tr)
        print(f"  R={sp.radius.flatten()[0].item():.4f}mm  (Gap 像方距离={gap_img})")

        # 几何真值：存活光线传感器落点在窗口内的占比
        pts = tr.sensor_pts  # (1,1,1,N,3)
        alive = tr.alive.flatten()
        xy = tr.sensor_tf.transform_points(pts, inverse=True)[..., :2]
        xy = xy[0, 0, 0]  # (N,2)
        c = rep.centers[0, 0]
        r = (xy - c).norm(dim=-1)
        geo = ((r < half) & alive).sum().item() / len(r)
        print(
            f"  几何包围（窗口内存活光线占比）={geo:.4f}  存活率={alive.float().mean().item():.4f}"
        )


if __name__ == "__main__":
    main()
