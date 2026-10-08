"""规范名词：一个术语的规范形式（canonical）与其全部别名。

配置键与参数名的查找、匹配、规范化一律经由 :class:`Noun`；全库词表见
:mod:`core.term`，业务代码中禁止散落硬编码字符串。
"""

import re
from collections.abc import Callable, Iterator, Mapping
from functools import partial
from typing import Any, Final, Self

from core.repr import render_line, styled

_MISSING: Final = object()  # resolve() 的"未提供 default"哨兵（区别于 default=None）


class Noun:
    """规范名词：``canonical`` 为规范形式，其余名称均为别名。

    相等性约定：与另一名词按规范形式比较，与字符串按"命中任意名称"比较。
    """

    __slots__ = ("_all_names", "_canonical", "_hash", "_name_index")

    def __init__(
        self,
        canonical: str,
        /,
        *aliases: str,
        normalize: Callable[[str], str] | None = None,
    ) -> None:
        if not canonical:
            raise ValueError("canonical name must be a non-empty string")

        if normalize is None:
            names = (canonical, *aliases)
        else:
            names = tuple(normalize(name) for name in (canonical, *aliases))

        deduplicated = tuple(dict.fromkeys(names))  # 去重并保持顺序
        self._canonical = deduplicated[0]
        self._all_names = deduplicated
        self._name_index = {name: i for i, name in enumerate(deduplicated)}
        self._hash = hash(self._canonical)

    @property
    def canonical(self) -> str:
        """名词的规范形式。"""
        return self._canonical

    @property
    def aliases(self) -> tuple[str, ...]:
        """名词的别名（不包括规范形式）。"""
        return self._all_names[1:]

    @property
    def all_names(self) -> tuple[str, ...]:
        """名词的全部名称（规范形式在前）。"""
        return self._all_names

    def __str__(self) -> str:
        return self._canonical

    def __repr__(self) -> str:
        return render_line(styled("Noun", ", ".join(map(repr, self._all_names))))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Noun):
            return self._canonical == other._canonical
        if isinstance(other, str):
            return other in self._name_index
        return NotImplemented

    def __contains__(self, name: str) -> bool:
        """*name* 是否命中本名词的任意名称。"""
        return name in self._name_index

    def __iter__(self) -> Iterator[str]:
        """按规范形式在前的顺序迭代全部名称。"""
        return iter(self._all_names)

    def __len__(self) -> int:
        """名称总数（规范形式 + 别名）。"""
        return len(self._all_names)

    def match(self, name: str) -> bool:
        """*name* 是否与本名词匹配（``name in self`` 的可读形式）。"""
        return name in self._name_index

    def canonicalize(self, name: str) -> str | None:
        """把 *name* 归一到规范形式；不匹配返回 ``None``。"""
        return self._canonical if name in self._name_index else None

    def resolve(self, mapping: Mapping[str, Any], default: Any = _MISSING) -> Any:
        """在 *mapping* 中按全部名称依序查找对应值。

        未命中时返回 *default*；未提供 *default* 则抛 ``KeyError``。
        """
        for name in self._all_names:
            if name in mapping:
                return mapping[name]
        if default is _MISSING:
            raise KeyError(
                f"missing required key: canonical={self._canonical!r}, names={self._all_names!r}"
            )
        return default

    def with_aliases(self, *aliases: str) -> Self:
        """返回追加别名后的新名词（既有名称已规范化，原样保留不再处理）。"""
        return type(self)(self._canonical, *self._all_names[1:], *aliases)

    @classmethod
    def for_key(cls, canonical: str, *aliases: str, sep: str = "_") -> Self:
        """创建适合作为配置键的名词：名称经 :func:`normalize_for_key` 规范化。"""
        return cls(canonical, *aliases, normalize=partial(normalize_for_key, sep=sep))


def normalize_for_key(name: str, sep: str = "_") -> str:
    """把名称规范化为键形式：空白与非字母数字字符折叠为 *sep*，首尾修剪。"""
    if not sep or not isinstance(sep, str):
        raise ValueError("separator must be a non-empty string")

    s = name.strip()
    s = re.sub(r"\s+", sep, s)  # 空白字符替换为分隔符
    s = re.sub(rf"[^A-Za-z0-9_{re.escape(sep)}-]+", sep, s)
    s = re.sub(rf"{re.escape(sep)}+", sep, s)  # 合并连续分隔符
    return s.strip(sep)
