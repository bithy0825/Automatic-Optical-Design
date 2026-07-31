"""二进制帧协议:Python 打包 ↔ TS 解码(web/src/shared/api.ts)的共享契约。

帧布局(小端)::

    [4B magic "AODV"][1B version][4B header_len][header JSON][payload]

header JSON = {"meta": ..., "sections": {name: {"offset", "shape", "dtype"}}},
dtype ∈ {"f32", "u8"},offset 相对 payload 起点。

对齐约束(JS TypedArray 视图要求):header 以空格填充使 (9 + header_len) % 8 == 0;
每段 payload 之后以 \\x00 填充至 4 字节边界,保证任何 f32 段的绝对偏移 % 4 == 0。
"""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np

MAGIC = b"AODV"
VERSION = 1

_DTYPES = {"f32": np.float32, "u8": np.uint8}
_NAMES = {v: k for k, v in _DTYPES.items()}


def pack(meta: dict[str, Any], sections: dict[str, np.ndarray]) -> bytes:
    """打包:*meta* 进 header JSON,*sections* 逐段拼入 payload。

    对齐:header 尾部以空格填充至 (9 + header_len) % 8 == 0;每段之后以
    \\x00 填充至 4 字节边界(JS TypedArray 视图的偏移对齐要求)。
    """
    header: dict[str, Any] = {"meta": meta, "sections": {}}
    payload = bytearray()
    for name, arr in sections.items():
        arr = np.ascontiguousarray(arr)
        try:
            dtype = _NAMES[arr.dtype.type]
        except KeyError:
            raise TypeError(f"section {name!r}: unsupported dtype {arr.dtype}") from None
        header["sections"][name] = {
            "offset": len(payload),
            "shape": list(arr.shape),
            "dtype": dtype,
        }
        payload += arr.tobytes()
        payload += b"\x00" * (-len(payload) % 4)
    hj = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hj += b" " * (-(9 + len(hj)) % 8)
    return MAGIC + bytes([VERSION]) + struct.pack("<I", len(hj)) + hj + bytes(payload)


def unpack(data: bytes) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """解包镜像(测试与调试用;生产解码在 TS 侧)。"""
    if data[:4] != MAGIC:
        raise ValueError("bad magic")
    if data[4] != VERSION:
        raise ValueError(f"unsupported version {data[4]}")
    (hlen,) = struct.unpack_from("<I", data, 5)
    header = json.loads(data[9 : 9 + hlen])
    base = 9 + hlen
    sections: dict[str, np.ndarray] = {}
    for name, desc in header["sections"].items():
        dt = np.dtype(_DTYPES[desc["dtype"]]).newbyteorder("<")
        count = int(np.prod(desc["shape"]))
        arr = np.frombuffer(data, dtype=dt, count=count, offset=base + desc["offset"])
        sections[name] = arr.reshape(desc["shape"])
    return header["meta"], sections
