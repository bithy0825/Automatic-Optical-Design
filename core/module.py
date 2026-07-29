from abc import ABC, abstractmethod
from typing import Any, ClassVar, Self
from collections.abc import Iterator, Mapping, Sequence
from itertools import chain

import torch
from torch import nn

from core.aliases import SystemBoolScalar, SystemLongScalar
from core.noun import Noun
from core.repr import render_tree, styled
from core.utils import fmt_param


class OpticalModule(nn.Module, ABC):
    mutable: ClassVar[tuple[Noun, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name in ("sort", "breed", "mutate", "clone", "where"):
            if name not in cls.__dict__:
                continue
            attr = cls.__dict__[name]
            if isinstance(attr, classmethod):
                # classmethod 对象不可调用，须包装其 __func__ 再重新包回
                setattr(cls, name, classmethod(torch.no_grad()(attr.__func__)))
            else:
                setattr(cls, name, torch.no_grad()(attr))

    # ------------------------------------------------------------------
    # 抽象契约
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """业务入口。语义由子类定义（如 ``Component.forward`` 消费并产出
        ``TraceFlow``，``Shape.forward`` 返回 ``TraceResult``）。"""

    @classmethod
    @abstractmethod
    def from_options(cls, population: int, options: Mapping[str, Any]) -> Self:
        """从配置映射构造 *population* 个个体的种群。"""

    @abstractmethod
    def clone(self) -> Self:
        """深拷贝：生成与本体互不干扰的独立个体（GA 演化所需）。"""

    # ------------------------------------------------------------------
    # 派生属性
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        for t in self.parameters():
            return t.device
        for t in self.buffers():
            return t.device
        return torch.get_default_device()

    @property
    def dtype(self) -> torch.dtype:
        for t in self.parameters():
            return t.dtype
        for t in self.buffers():
            if t.is_floating_point():
                return t.dtype
        return torch.get_default_dtype()

    @property
    def population(self) -> int:
        """首个参数/buffer 的批量维大小（P）；无参数时抛错。"""
        for t in self.parameters():
            return t.shape[0]
        for t in self.buffers():
            return t.shape[0]
        raise RuntimeError(f"{type(self).__name__} has no batched parameters")

    # ------------------------------------------------------------------
    # GA 演化操作（默认实现，子类按需覆盖）
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sort(self, order: SystemLongScalar) -> None:
        """按 *order* 重排所有批量张量（含子模块）。"""
        for _name, t in self._batched_tensors():
            t.copy_(t.index_select(0, order))

    @torch.no_grad()
    def breed(self, topk: int) -> None:
        """用前 *topk* 个精英滚动复制填充整个种群。"""
        assert 0 < topk <= self.population, (
            f"topk must be in (0, {self.population}], got {topk}"
        )
        idx = torch.arange(self.population - topk, device=self.device).remainder(topk)
        for _name, t in self._batched_tensors():
            t[topk:].copy_(t[:topk][idx])

    @torch.no_grad()
    def mutate(self, indices: SystemLongScalar, options: Mapping[str, Any]) -> None:
        """对指定索引的个体做高斯扰动（按 ``mutable`` 词表逐项取标准差，
        缺省或为零则跳过）。"""
        for noun in self.mutable:
            std = noun.resolve(options, default=0.0)
            if std == 0.0:
                continue
            tensor = getattr(self, noun.canonical)
            noise = torch.randn_like(tensor[indices]).mul(std)
            tensor.index_put_((indices,), noise, accumulate=True)

    @classmethod
    @torch.no_grad()
    def where(cls, mask: SystemBoolScalar, new: Self, old: Self) -> Self:
        """种群级个体选择：``mask=True`` 取 *new*，``False`` 取 *old*，两边均不被修改。

        默认实现克隆 *new* 后把 ``~mask`` 行从 *old* 逐批量张量回写。子类可覆盖为
        逐字段 ``torch.where`` 直构（镜像各自 ``clone`` 的字段清单），约定：非张量
        共享配置（求解器选项、照明配置等）从 *new* 继承；多维参数（如 ``(P, N)``
        的 Alpha）对 mask 自行升维广播。
        """
        OpticalModule._check_operands(mask, new, old)
        merged = new.clone()
        reject = (~mask).nonzero().squeeze(-1)
        if reject.numel():
            mine = list(merged._batched_tensors())
            theirs = list(old._batched_tensors())
            assert [n for n, _ in mine] == [n for n, _ in theirs], "module trees differ"
            for (_, t), (_, o) in zip(mine, theirs):
                t.index_copy_(0, reject, o.index_select(0, reject))
        return merged

    @staticmethod
    def _check_operands(
        mask: SystemBoolScalar, new: "OpticalModule", old: "OpticalModule"
    ) -> None:
        """``where`` 操作数校验：同类型、同种群、mask 为 ``(P,)`` bool。"""
        if type(new) is not type(old):
            raise TypeError(
                f"where: {type(new).__name__} vs {type(old).__name__}"
            )
        if new.population != old.population:
            raise ValueError(
                f"where: population {new.population} vs {old.population}"
            )
        if mask.dtype != torch.bool or mask.shape != (new.population,):
            raise ValueError(
                f"where: mask must be a ({new.population},) bool tensor, "
                f"got shape={tuple(mask.shape)}, dtype={mask.dtype}"
            )

    def __getitem__(self, key: Noun) -> torch.Tensor:
        return getattr(self, key.canonical)

    # ------------------------------------------------------------------
    # 打印
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return render_tree(self)

    def _label(self) -> str:
        """树节点标签行（子类按需覆盖，如 Sequential 的头部信息）。"""
        return styled(type(self).__name__)

    def _params(self) -> Iterator[str]:
        """树节点名下的参数行：每个注册参数/buffer 一行（子类按需覆盖）。"""
        for name, t in chain(
            self.named_parameters(recurse=False),
            self.named_buffers(recurse=False),
        ):
            if t.ndim == 1:
                yield f"{name}={fmt_param(t)}"
            else:
                yield f"{name}={tuple(t.shape)}"

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _reorder(self, name: str | Noun, order: SystemLongScalar) -> None:
        key = name.canonical if isinstance(name, Noun) else name
        val = getattr(self, key)
        val.copy_(val.index_select(0, order))

    def _batched_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        """递归产生批量维为 P 的 (名称, 张量)（含子模块的参数与 buffer）。"""
        P = self.population
        for name, param in self.named_parameters():
            if param.shape[0] == P:
                yield name, param
        for name, buffer in self.named_buffers():
            if buffer.shape[0] == P:
                yield name, buffer


def init_param(
    parent: nn.Module,
    name: str | Noun,
    value: float | Sequence[float] | torch.Tensor,
    trainable: bool = False,
) -> torch.Tensor:
    key = name.canonical if isinstance(name, Noun) else name

    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    else:
        first_param = next(parent.parameters(), None)
        device = (
            first_param.device
            if first_param is not None
            else torch.get_default_device()
        )
        tensor = torch.tensor(value, device=device)

    if trainable:
        param = nn.Parameter(tensor)
        parent.register_parameter(key, param)
        return param
    parent.register_buffer(key, tensor)
    return tensor
