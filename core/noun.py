import re
from collections.abc import Callable, Iterator, Mapping
from functools import partial
from typing import Any, Final, Self

from core.repr import render_line, styled

_MISSING: Final = object()  # resolve() 的"未提供 default"哨兵（区别于 default=None）


class Noun:
    __slots__ = ("_all_names", "_canonical", "_hash", "_name_index")

    def __init__(
        self,
        canonical: str,
        /,
        *aliases: str,
        normalize: Callable[[str], str] | None = None,
    ) -> None:
        """
        创建一个名词对象。

        :param canonical: 名词的规范形式。
        :param aliases: 名词的别名。
        :param normalize: 可选的规范化函数，用于标准化输入字符串。
        """
        if not canonical:
            raise ValueError("Canonical name must be a non-empty string.")

        if normalize is None:
            names = (canonical, *aliases)
        else:
            names = tuple(normalize(name) for name in (canonical, *aliases))

        deduplicated_names = tuple(dict.fromkeys(names))  # 去重并保持顺序
        self._canonical = deduplicated_names[0]
        self._all_names = deduplicated_names
        self._name_index = {name: i for i, name in enumerate(deduplicated_names)}
        self._hash = hash(self._canonical)

    @property
    def canonical(self) -> str:
        """返回名词的规范形式。"""
        return self._canonical

    @property
    def aliases(self) -> tuple[str, ...]:
        """返回名词的别名（不包括规范形式）。"""
        return self._all_names[1:]

    @property
    def all_names(self) -> tuple[str, ...]:
        """返回名词的所有名称，包括规范形式和别名。"""
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
        """检查给定的名称是否是名词的规范形式或别名之一。"""
        return name in self._name_index

    def __iter__(self) -> Iterator[str]:
        """返回名词的所有名称的迭代器。"""
        return iter(self._all_names)

    def __len__(self) -> int:
        """返回名词的名称总数，包括规范形式和别名。"""
        return len(self._all_names)

    def match(self, name: str) -> bool:
        """检查给定的名称是否与名词的规范形式或别名匹配。"""
        return name in self._name_index

    def canonicalize(self, name: str) -> str | None:
        """将给定的名称规范化为名词的规范形式，如果不匹配则返回 None。"""
        return self._canonical if name in self._name_index else None

    def resolve(self, mapping: Mapping[str, Any], default: Any = _MISSING) -> Any:
        """
        在给定的映射中查找名词的规范形式或别名对应的值。

        :param mapping: 要查找的映射。
        :param default: 如果未找到匹配项，则返回的默认值。如果未提供，则引发 KeyError。
        :return: 映射中对应的值，或默认值（如果提供）。
        :raises KeyError: 如果未找到匹配项且未提供默认值。
        """
        for name in self._all_names:
            if name in mapping:
                return mapping[name]
        if default is _MISSING:
            raise KeyError(
                f"Missing required key: canonical={self._canonical!r}, "
                f"names={self._all_names!r}"
            )
        return default

    def with_aliases(self, *aliases: str) -> Self:
        """
        返回一个新的 Noun 对象，包含当前对象的规范形式和别名，以及额外提供的别名。

        :param aliases: 额外的别名。
        :return: 新的 Noun 对象。
        """
        return type(self)(self._canonical, *self._all_names[1:], *aliases)

    @classmethod
    def for_key(cls, canonical: str, *aliases: str, sep: str = "_") -> Self:
        """
        创建一个适合作为键的 Noun 对象，使用给定的分隔符规范化名称。

        :param canonical: 名词的规范形式。
        :param aliases: 名词的别名。
        :param sep: 用于替换非字母数字字符的分隔符，默认为下划线。
        :return: 新的 Noun 对象。
        """
        normalize = partial(normalize_for_key, sep=sep)
        return cls(canonical, *aliases, normalize=normalize)


def normalize_for_key(name: str, sep: str = "_") -> str:
    """
    将给定的名称规范化为适合作为键的形式。

    :param name: 要规范化的名称。
    :param sep: 用于替换非字母数字字符的分隔符，默认为下划线。
    :return: 规范化后的名称。
    """
    if not sep or not isinstance(sep, str):
        raise ValueError("Separator must be a non-empty string.")

    s = name.strip()
    s = re.sub(r"\s+", sep, s)  # 将空白字符替换为分隔符
    s = re.sub(rf"[^A-Za-z0-9_{re.escape(sep)}-]+", sep, s)
    s = re.sub(rf"{re.escape(sep)}+", sep, s)  # 合并连续的分隔符
    s = s.strip(sep)  # 去除开头和结尾的分隔符
    return s
