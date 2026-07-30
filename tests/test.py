"""Tests for clone, sort, breed, mutate, and where on Sequential optical systems.

Core requirements:
- ``clone()`` must produce a completely independent individual: no tensor memory
  is shared with the original, and every owned sub-object (shape, material, …)
  is recursively deep-copied.
- ``sort``, ``breed``, and ``mutate`` must operate correctly on cloned objects.
- Mutating or sorting a clone must never affect the original (and vice versa).
- ``where(mask, new, old)`` must merge two systems row-wise without touching
  either operand, and rebuild the material reference chain.

Run:
    python test.py          # or  pytest test.py
"""

import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from component import Sequential
from component.gap import Gap
from component.refractor import Refractor
from component.source import InfiniteSource
from core import OpticalModule, term
from optimization import build_sequential
from shape import Asphere

# ═══════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════

CONFIG_PATH = Path(__file__).resolve().parents[1] / "demo" / "config.toml"
POPULATION = 8


def _build() -> "Sequential":
    return build_sequential(str(CONFIG_PATH))


def _reload_component_blocks() -> list[dict[str, Any]]:
    """Reload raw component blocks (as passed to Sequential.from_options)."""
    with open(CONFIG_PATH, "rb") as f:
        cfg = tomllib.load(f)
    return list(cfg["component"])


def _all_tensors(
    module: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Collect all (dotted_name → tensor) including submodules."""
    tensors: dict[str, torch.Tensor] = {}
    for n, p in module.named_parameters():
        tensors[n] = p
    for n, b in module.named_buffers():
        tensors[n] = b
    return tensors


def _device(module: torch.nn.Module) -> torch.device:
    """Return the device of the first tensor in *module*."""
    for p in module.parameters():
        return p.device
    for b in module.buffers():
        return b.device
    return torch.get_default_device()


# ═══════════════════════════════════════════════════════════════════════
# 1. clone — complete independence
# ═══════════════════════════════════════════════════════════════════════


def test_clone_is_different_object():
    """clone() returns a new Python object, not the same id."""
    s = _build()
    ss = s.clone()
    assert s is not ss, "clone() returned the same object"


def test_clone_no_shared_tensor_memory():
    """Every tensor (param + buffer) in the clone must live at a different
    address from the corresponding tensor in the original."""
    s = _build()
    ss = s.clone()

    s_tensors = _all_tensors(s)
    ss_tensors = _all_tensors(ss)

    common = set(s_tensors) & set(ss_tensors)
    assert common, "no common tensor names between original and clone"

    for name in sorted(common):
        p_s = s_tensors[name].data_ptr()
        p_ss = ss_tensors[name].data_ptr()
        assert p_s != p_ss, (
            f"Tensor {name!r} shares memory: "
            f"original={hex(p_s)}, clone={hex(p_ss)}"
        )


def test_clone_values_equal():
    """After clone, every tensor must have the same value (just not the same
    storage)."""
    s = _build()
    ss = s.clone()

    s_tensors = _all_tensors(s)
    ss_tensors = _all_tensors(ss)

    for name in sorted(s_tensors):
        assert torch.equal(s_tensors[name], ss_tensors[name]), (
            f"Tensor {name!r} differs between original and clone"
        )


def test_clone_material_ref_chain_rebuilt():
    """After clone + rebind, each Refractor's incident MaterialRef must point
    to the clone's own upstream Material — NOT the original's."""
    s = _build()
    ss = s.clone()

    # Refractors are at indices 2, 4, 6, 8.
    # Their incidents should point to: 0, 2, 4, 6 respectively (in ss's chain).
    incident_chain = [(2, 0), (4, 2), (6, 4), (8, 6)]

    for refr_idx, upstream_idx in incident_chain:
        refr_s = s[refr_idx]
        refr_ss = ss[refr_idx]
        upstream_s = s[upstream_idx]
        upstream_ss = ss[upstream_idx]

        assert isinstance(refr_s, Refractor)
        assert isinstance(refr_ss, Refractor)

        # Clone's incident must reference clone's own upstream transmitted
        assert (
            refr_ss.incident._material is upstream_ss.transmitted
        ), (
            f"ss[{refr_idx}].incident does NOT point to ss[{upstream_idx}].transmitted"
        )

        # Clone's incident must NOT reference original's upstream
        assert (
            refr_ss.incident._material is not upstream_s.transmitted
        ), (
            f"ss[{refr_idx}].incident still points to s[{upstream_idx}].transmitted "
            f"(shared original chain!)"
        )

        # Original's incident must reference original's upstream
        assert (
            refr_s.incident._material is upstream_s.transmitted
        ), (
            f"s[{refr_idx}].incident does NOT point to s[{upstream_idx}].transmitted"
        )


def test_clone_modify_does_not_affect_original_sort():
    """Sorting the clone must leave the original completely unchanged."""
    s = _build()
    ss = s.clone()

    snap_s = _all_tensors(s)

    rev = torch.arange(POPULATION - 1, -1, -1, device=_device(ss), dtype=torch.long)
    ss.sort_(rev)

    for name, t_before in snap_s.items():
        t_after = dict(_all_tensors(s))[name]
        assert torch.equal(t_before, t_after), (
            f"Original's tensor {name!r} changed after clone was sorted"
        )


def test_clone_modify_does_not_affect_original_breed():
    """Breeding the clone must leave the original completely unchanged."""
    s = _build()
    ss = s.clone()

    snap_s = _all_tensors(s)

    ss.breed_(topk=4)

    for name, t_before in snap_s.items():
        t_after = dict(_all_tensors(s))[name]
        assert torch.equal(t_before, t_after), (
            f"Original's tensor {name!r} changed after clone was bred"
        )


def test_clone_modify_does_not_affect_original_mutate():
    """Mutating the clone must leave the original completely unchanged."""
    s = _build()
    ss = s.clone()

    snap_s = _all_tensors(s)
    blocks = _reload_component_blocks()
    indices = torch.tensor([0, 2, 5], device=_device(ss), dtype=torch.long)

    ss.mutate_(indices, blocks)

    for name, t_before in snap_s.items():
        t_after = dict(_all_tensors(s))[name]
        assert torch.equal(t_before, t_after), (
            f"Original's tensor {name!r} changed after clone was mutated"
        )


def test_original_modify_does_not_affect_clone():
    """Modifying the original must leave a previously-taken clone unchanged."""
    s = _build()
    ss = s.clone()

    snap_ss = _all_tensors(ss)

    rev = torch.arange(POPULATION - 1, -1, -1, device=_device(s), dtype=torch.long)
    s.sort_(rev)

    for name, t_before in snap_ss.items():
        t_after = dict(_all_tensors(ss))[name]
        assert torch.equal(t_before, t_after), (
            f"Clone's tensor {name!r} changed after original was sorted"
        )


# ═══════════════════════════════════════════════════════════════════════
# 2. sort — permutation correctness
# ═══════════════════════════════════════════════════════════════════════


def test_sort_applies_correct_permutation():
    """After sort(order), every batched tensor's dim-0 must equal the
    permuted pre-sort values."""
    s = _build()
    P = s.population

    # Record pre-sort values for all batched tensors
    pre: dict[str, torch.Tensor] = {}
    for name, t in _all_tensors(s).items():
        if t.ndim >= 1 and t.shape[0] == P:
            pre[name] = t.clone()

    # Simple shift-forward: [1,2,3,...,P-1,0]
    order = torch.arange(P, device=_device(s), dtype=torch.long)
    order = (order + 1) % P
    s.sort_(order)

    for name, t_pre in pre.items():
        t_post = dict(_all_tensors(s))[name]
        expected = t_pre.index_select(0, order)
        assert torch.equal(t_post, expected), (
            f"sort permutation incorrect for {name!r}"
        )


def test_sort_original_and_clone_independent():
    """Sort original and clone with DIFFERENT orders → each gets its own
    permutation without cross-talk."""
    s = _build()
    ss = s.clone()
    P = s.population
    dev = _device(s)

    snap_s_before: dict[str, torch.Tensor] = {}
    for name, t in _all_tensors(s).items():
        if t.ndim >= 1 and t.shape[0] == P:
            snap_s_before[name] = t.clone()
    snap_ss_before: dict[str, torch.Tensor] = {}
    for name, t in _all_tensors(ss).items():
        if t.ndim >= 1 and t.shape[0] == P:
            snap_ss_before[name] = t.clone()

    order_s = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], device=dev, dtype=torch.long)
    order_ss = torch.tensor([1, 0, 2, 3, 4, 5, 6, 7], device=dev, dtype=torch.long)

    s.sort_(order_s)
    ss.sort_(order_ss)

    # s must match s_before permuted by order_s
    for name, t_pre in snap_s_before.items():
        t_post = dict(_all_tensors(s))[name]
        if t_pre.ndim >= 1 and t_pre.shape[0] == P:
            assert torch.equal(t_post, t_pre.index_select(0, order_s)), (
                f"Original {name!r} was not correctly permuted"
            )

    # ss must match ss_before permuted by order_ss
    for name, t_pre in snap_ss_before.items():
        t_post = dict(_all_tensors(ss))[name]
        if t_pre.ndim >= 1 and t_pre.shape[0] == P:
            assert torch.equal(t_post, t_pre.index_select(0, order_ss)), (
                f"Clone {name!r} was not correctly permuted"
            )


# ═══════════════════════════════════════════════════════════════════════
# 3. breed — top-k elite replication
# ═══════════════════════════════════════════════════════════════════════


def test_breed_preserves_topk():
    """After breed(topk=K), population positions 0..K-1 must be unchanged."""
    s = _build()
    P = s.population
    K = 4

    pre = {
        name: t.clone()
        for name, t in _all_tensors(s).items()
        if t.ndim >= 1 and t.shape[0] == P
    }

    s.breed_(K)

    for name, t_pre in pre.items():
        t_post = dict(_all_tensors(s))[name]
        assert torch.equal(t_post[:K], t_pre[:K]), (
            f"Top-{K} of {name!r} changed after breed (should be preserved)"
        )


def test_breed_fills_remainder_cyclically():
    """After breed(topk=K), position i (i ≥ K) must equal position
    (i-K) % K of the pre-breed elite."""
    s = _build()
    P = s.population
    K = 4

    pre = {
        name: t.clone()
        for name, t in _all_tensors(s).items()
        if t.ndim >= 1 and t.shape[0] == P
    }

    s.breed_(K)

    idx = torch.arange(P - K, device=_device(s)).remainder(K)
    for name, t_pre in pre.items():
        t_post = dict(_all_tensors(s))[name]
        expected = t_pre[:K][idx]
        assert torch.equal(t_post[K:], expected), (
            f"breed fill incorrect for {name!r}"
        )


def test_breed_topk_2():
    """Edge case: breed with topk=2."""
    s = _build()
    P = s.population
    K = 2

    pre = {
        name: t.clone()
        for name, t in _all_tensors(s).items()
        if t.ndim >= 1 and t.shape[0] == P
    }

    s.breed_(K)

    # top-K unchanged
    for name, t_pre in pre.items():
        t_post = dict(_all_tensors(s))[name]
        assert torch.equal(t_post[:K], t_pre[:K]), (
            f"Top-{K} changed after breed(topk={K}) for {name!r}"
        )

    # remainder filled cyclically
    idx = torch.arange(P - K, device=_device(s)).remainder(K)
    for name, t_pre in pre.items():
        t_post = dict(_all_tensors(s))[name]
        expected = t_pre[:K][idx]
        assert torch.equal(t_post[K:], expected), (
            f"breed(topk={K}) fill incorrect for {name!r}"
        )


def test_breed_original_and_clone_independent():
    """Breed original and clone independently → neither affects the other."""
    s = _build()
    ss = s.clone()
    P = s.population

    pre_s = {
        name: t.clone()
        for name, t in _all_tensors(s).items()
        if t.ndim >= 1 and t.shape[0] == P
    }
    pre_ss = {
        name: t.clone()
        for name, t in _all_tensors(ss).items()
        if t.ndim >= 1 and t.shape[0] == P
    }

    s.breed_(3)
    ss.breed_(5)

    # s after breed(3): check against s pre
    idx_s = torch.arange(P - 3, device=_device(s)).remainder(3)
    for name, t_pre in pre_s.items():
        t_post = dict(_all_tensors(s))[name]
        assert torch.equal(t_post[:3], t_pre[:3]), f"topk shift in original {name!r}"
        assert torch.equal(t_post[3:], t_pre[:3][idx_s]), f"fill shift in original {name!r}"

    # ss after breed(5): check against ss pre
    idx_ss = torch.arange(P - 5, device=_device(ss)).remainder(5)
    for name, t_pre in pre_ss.items():
        t_post = dict(_all_tensors(ss))[name]
        assert torch.equal(t_post[:5], t_pre[:5]), f"topk shift in clone {name!r}"
        assert torch.equal(t_post[5:], t_pre[:5][idx_ss]), f"fill shift in clone {name!r}"


# ═══════════════════════════════════════════════════════════════════════
# 4. mutate — targeted Gaussian perturbation
# ═══════════════════════════════════════════════════════════════════════


def _gather_mutable_tensors(
    module: torch.nn.Module, P: int
) -> dict[str, torch.Tensor]:
    """Collect all batched tensors (dim-0 == P) from *module*.

    We scope to batched tensors because non-batched tensors (e.g. singleton
    database buffers, solver state) are never mutated by GA operations.
    """
    result: dict[str, torch.Tensor] = {}
    for name, t in _all_tensors(module).items():
        if t.ndim >= 1 and t.shape[0] == P:
            result[name] = t
    return result


def test_mutate_non_targeted_unchanged():
    """Only the specified indices should change; others must stay identical."""
    s = _build()
    P = s.population

    pre = {name: t.clone() for name, t in _gather_mutable_tensors(s, P).items()}

    blocks = _reload_component_blocks()
    target_idx = torch.tensor([2, 5], device=_device(s), dtype=torch.long)
    s.mutate_(target_idx, blocks)

    # Build a boolean mask of non-targeted population indices
    all_idx = torch.arange(P, device=_device(s))
    mask = ~torch.isin(all_idx, target_idx)  # True = not targeted

    for name, t_pre in pre.items():
        t_post = dict(_gather_mutable_tensors(s, P))[name]
        assert torch.equal(t_post[mask], t_pre[mask]), (
            f"Non-targeted indices of {name!r} changed during mutate!"
        )


def test_mutate_targeted_changed():
    """Mutated indices must differ from before (Gaussian noise with non-zero
    std on at least some parameters)."""
    s = _build()
    P = s.population

    pre = {name: t.clone() for name, t in _gather_mutable_tensors(s, P).items()}

    blocks = _reload_component_blocks()
    target_idx = torch.tensor([0, 2, 4, 6], device=_device(s), dtype=torch.long)
    s.mutate_(target_idx, blocks)

    any_changed = False
    for name, t_pre in pre.items():
        t_post = dict(_gather_mutable_tensors(s, P))[name]
        if not torch.equal(t_post[target_idx], t_pre[target_idx]):
            any_changed = True
            break

    assert any_changed, (
        "No tensor changed on targeted indices — mutate had no effect. "
        "Check that at least one component has non-zero mutation std."
    )


def test_mutate_single_individual():
    """Mutating a single individual must change only that one."""
    s = _build()
    P = s.population

    pre = {name: t.clone() for name, t in _gather_mutable_tensors(s, P).items()}

    blocks = _reload_component_blocks()
    single = torch.tensor([3], device=_device(s), dtype=torch.long)
    s.mutate_(single, blocks)

    mask = torch.ones(P, dtype=torch.bool, device=_device(s))
    mask[3] = False  # False = was targeted

    for name, t_pre in pre.items():
        t_post = dict(_gather_mutable_tensors(s, P))[name]
        assert torch.equal(t_post[mask], t_pre[mask]), (
            f"Non-targeted indices of {name!r} changed when only index 3 was mutated"
        )


def test_mutate_original_and_clone_independent():
    """Mutate original and clone on different indices → each only sees its
    own perturbation."""
    s = _build()
    ss = s.clone()
    P = s.population
    dev = _device(s)

    pre_s = {name: t.clone() for name, t in _gather_mutable_tensors(s, P).items()}
    pre_ss = {name: t.clone() for name, t in _gather_mutable_tensors(ss, P).items()}

    blocks = _reload_component_blocks()
    idx_s = torch.tensor([0, 1], device=dev, dtype=torch.long)
    idx_ss = torch.tensor([2, 3], device=dev, dtype=torch.long)

    s.mutate_(idx_s, blocks)
    ss.mutate_(idx_ss, blocks)

    # s: only indices 0,1 may change; the rest unchanged
    mask_s_keep = torch.ones(P, dtype=torch.bool, device=dev)
    mask_s_keep[idx_s] = False
    for name, t_pre in pre_s.items():
        t_post = dict(_gather_mutable_tensors(s, P))[name]
        assert torch.equal(t_post[mask_s_keep], t_pre[mask_s_keep]), (
            f"Original {name!r} changed on indices {idx_s.tolist()}"
        )

    # ss: only indices 2,3 may change; the rest unchanged
    mask_ss_keep = torch.ones(P, dtype=torch.bool, device=dev)
    mask_ss_keep[idx_ss] = False
    for name, t_pre in pre_ss.items():
        t_post = dict(_gather_mutable_tensors(ss, P))[name]
        assert torch.equal(t_post[mask_ss_keep], t_pre[mask_ss_keep]), (
            f"Clone {name!r} changed on indices {idx_ss.tolist()}"
        )


def test_mutate_without_clone_then_clone_isolation():
    """Mutate original BEFORE cloning → clone gets mutated values, but
    subsequent mutations on clone do NOT back-propagate to original."""
    s = _build()

    blocks = _reload_component_blocks()
    idx = torch.tensor([0, 1], device=_device(s), dtype=torch.long)
    s.mutate_(idx, blocks)

    ss = s.clone()
    P = s.population
    dev = _device(s)

    # Clone should have the mutated values
    s_tensors = _gather_mutable_tensors(s, P)
    ss_tensors = _gather_mutable_tensors(ss, P)
    for name in s_tensors:
        assert torch.equal(s_tensors[name], ss_tensors[name]), (
            f"Clone {name!r} doesn't match mutated original"
        )

    # Now mutate clone on different indices
    snap_s_after_clone = {name: t.clone() for name, t in s_tensors.items()}
    idx2 = torch.tensor([4, 5], device=dev, dtype=torch.long)
    ss.mutate_(idx2, blocks)

    # Original must be unchanged by clone's mutation
    for name, t_snap in snap_s_after_clone.items():
        t_now = dict(_gather_mutable_tensors(s, P))[name]
        assert torch.equal(t_now, t_snap), (
            f"Original {name!r} changed after clone was mutated post-clone"
        )


# ═══════════════════════════════════════════════════════════════════════
# 5. end-to-end: clone → sort → breed → mutate chain
# ═══════════════════════════════════════════════════════════════════════


def test_pipeline_clone_sort_breed_mutate():
    """Full GA pipeline on a clone: clone → sort → breed → mutate.
    Original must remain entirely untouched."""
    s = _build()
    P = s.population
    dev = _device(s)

    snap_s = {name: t.clone() for name, t in _gather_mutable_tensors(s, P).items()}

    # ── 1. clone ──
    ss = s.clone()
    assert s is not ss
    for name, t_s in snap_s.items():
        t_ss = dict(_gather_mutable_tensors(ss, P))[name]
        assert t_ss.data_ptr() != t_s.data_ptr(), f"{name!r} shares memory"
        assert torch.equal(t_s, t_ss), f"{name!r} values differ after clone"

    # ── 2. sort clone ──
    order = torch.randperm(P, device=dev, dtype=torch.long)
    pre_sort_ss = {name: t.clone() for name, t in _gather_mutable_tensors(ss, P).items()}
    ss.sort_(order)
    for name, t_pre in pre_sort_ss.items():
        t_post = dict(_gather_mutable_tensors(ss, P))[name]
        assert torch.equal(t_post, t_pre.index_select(0, order)), (
            f"sort failed for clone {name!r}"
        )

    # ── 3. breed clone ──
    K = 4
    pre_breed_ss = {name: t.clone() for name, t in _gather_mutable_tensors(ss, P).items()}
    ss.breed_(K)
    idx = torch.arange(P - K, device=dev).remainder(K)
    for name, t_pre in pre_breed_ss.items():
        t_post = dict(_gather_mutable_tensors(ss, P))[name]
        assert torch.equal(t_post[:K], t_pre[:K]), f"breed topk changed {name!r}"
        assert torch.equal(t_post[K:], t_pre[:K][idx]), f"breed fill wrong {name!r}"

    # ── 4. mutate clone ──
    blocks = _reload_component_blocks()
    target = torch.tensor([2, 7], device=dev, dtype=torch.long)
    ss.mutate_(target, blocks)

    # ── 5. original must be completely unchanged ──
    for name, t_snap in snap_s.items():
        t_now = dict(_gather_mutable_tensors(s, P))[name]
        assert torch.equal(t_now, t_snap), (
            f"Original {name!r} was modified during clone pipeline"
        )


# ═══════════════════════════════════════════════════════════════════════
# 6. where — SA 接受/拒绝的种群级合并
# ═══════════════════════════════════════════════════════════════════════

MASK_PATTERN = [True, False, True, True, False, False, True, False]


def _mask(device: torch.device) -> torch.Tensor:
    return torch.tensor(MASK_PATTERN, dtype=torch.bool, device=device)


def _expected_where(
    mask: torch.Tensor, t_new: torch.Tensor, t_old: torch.Tensor
) -> torch.Tensor:
    """期望值：mask 升维到被择张量的 ndim 后 torch.where。"""
    m = mask.view(mask.shape[0], *([1] * (t_new.ndim - 1)))
    return torch.where(m, t_new, t_old)


def _diverged_pair() -> tuple["Sequential", "Sequential"]:
    """(new, old)：old 为快照，new 全体变异产生分歧。"""
    s = _build()
    old = s.clone()
    blocks = _reload_component_blocks()
    all_idx = torch.arange(s.population, device=_device(s), dtype=torch.long)
    s.mutate_(all_idx, blocks)
    return s, old


def test_where_non_destructive():
    """where 不得修改 new 和 old 的任何张量。"""
    s, old = _diverged_pair()
    mask = _mask(_device(s))
    snap_s = {n: t.clone() for n, t in _all_tensors(s).items()}
    snap_old = {n: t.clone() for n, t in _all_tensors(old).items()}

    Sequential.where(mask, s, old)

    for name, t in snap_s.items():
        assert torch.equal(dict(_all_tensors(s))[name], t), (
            f"new's tensor {name!r} changed during where"
        )
    for name, t in snap_old.items():
        assert torch.equal(dict(_all_tensors(old))[name], t), (
            f"old's tensor {name!r} changed during where"
        )


def test_where_row_correctness():
    """merged 的每个批量张量逐行等于 torch.where(mask, new, old)。"""
    s, old = _diverged_pair()
    P = s.population
    mask = _mask(_device(s))
    merged = Sequential.where(mask, s, old)

    t_new = _gather_mutable_tensors(s, P)
    t_old = _gather_mutable_tensors(old, P)
    t_merged = _gather_mutable_tensors(merged, P)
    assert set(t_merged) == set(t_new) == set(t_old), "tensor topology changed"

    for name in t_merged:
        expected = _expected_where(mask, t_new[name], t_old[name])
        assert torch.equal(t_merged[name], expected), (
            f"where row selection wrong for {name!r}"
        )


def test_where_material_names():
    """玻璃选择逐行来自正确的一侧（source + 四个 refractor 的 transmitted）。"""
    s, old = _diverged_pair()
    mask = _mask(_device(s))
    merged = Sequential.where(mask, s, old)

    for idx in (0, 2, 4, 6, 8):
        names_new = s[idx].transmitted.names()
        names_old = old[idx].transmitted.names()
        names_m = merged[idx].transmitted.names()
        for i in range(s.population):
            expect = names_new[i] if mask[i] else names_old[i]
            assert names_m[i] == expect, (
                f"component {idx} individual {i}: expected {expect}, got {names_m[i]}"
            )


def test_where_all_true():
    """mask 全 True：merged 逐张量等于 new。"""
    s, old = _diverged_pair()
    P = s.population
    mask = torch.ones(P, dtype=torch.bool, device=_device(s))
    merged = Sequential.where(mask, s, old)

    for name, t in _gather_mutable_tensors(merged, P).items():
        assert torch.equal(t, dict(_gather_mutable_tensors(s, P))[name]), (
            f"all-True where differs from new for {name!r}"
        )


def test_where_all_false():
    """mask 全 False：merged 逐张量等于 old。"""
    s, old = _diverged_pair()
    P = s.population
    mask = torch.zeros(P, dtype=torch.bool, device=_device(s))
    merged = Sequential.where(mask, s, old)

    for name, t in _gather_mutable_tensors(merged, P).items():
        assert torch.equal(t, dict(_gather_mutable_tensors(old, P))[name]), (
            f"all-False where differs from old for {name!r}"
        )


def test_where_incident_rebound():
    """merged 的 MaterialRef 链指向 merged 自己的上游，而非 new/old 的。"""
    s, old = _diverged_pair()
    merged = Sequential.where(_mask(_device(s)), s, old)

    for refr_idx, up_idx in [(2, 0), (4, 2), (6, 4), (8, 6)]:
        refr = merged[refr_idx]
        assert isinstance(refr, Refractor)
        assert refr.incident._material is merged[up_idx].transmitted, (
            f"merged[{refr_idx}].incident not rebound to merged[{up_idx}]"
        )
        assert refr.incident._material is not s[up_idx].transmitted
        assert refr.incident._material is not old[up_idx].transmitted


def test_where_forward_runs():
    """合并后的系统可以正常追迹。"""
    s, old = _diverged_pair()
    merged = Sequential.where(_mask(_device(s)), s, old)
    flow = merged()
    assert flow.rays.points.shape[0] == s.population


def test_where_type_mismatch_raises():
    """两个操作数类型不同 → TypeError；链长不同 → ValueError。"""
    s = _build()
    mask = _mask(_device(s))

    try:
        Gap.where(mask, s[1], s[2])  # Gap vs Refractor
        raise AssertionError("type mismatch not caught")
    except TypeError:
        pass

    short = Sequential(s[0].clone(), s[1].clone())
    try:
        Sequential.where(mask, s, short)  # 元件数不同
        raise AssertionError("length mismatch not caught")
    except ValueError:
        pass


def test_where_bad_mask_raises():
    """mask 非 bool 或长度不符 → ValueError。"""
    s = _build()
    old = s.clone()
    P = s.population
    dev = _device(s)

    try:
        Sequential.where(torch.arange(P, device=dev), s, old)
        raise AssertionError("long mask not caught")
    except ValueError:
        pass

    try:
        Sequential.where(torch.ones(P - 1, dtype=torch.bool, device=dev), s, old)
        raise AssertionError("short mask not caught")
    except ValueError:
        pass


def test_where_population_mismatch_raises():
    """两个操作数 population 不同 → ValueError。"""
    g8 = Gap(torch.full((8,), 5.0))
    g3 = Gap(torch.full((3,), 5.0))
    try:
        Gap.where(torch.ones(8, dtype=torch.bool), g8, g3)
        raise AssertionError("population mismatch not caught")
    except ValueError:
        pass


def test_where_base_default_matches_bespoke():
    """基类默认实现（clone + 行回写）与子类直构结果一致。"""
    s, old = _diverged_pair()
    mask = _mask(_device(s))
    g_new, g_old = s[3], old[3]

    via_base = OpticalModule.where(mask, g_new, g_old)
    via_bespoke = Gap.where(mask, g_new, g_old)
    assert torch.equal(via_base.Thickness, via_bespoke.Thickness)


def test_where_asphere_alpha_broadcast():
    """(P, N) 的 Alpha：mask 升维广播，逐行选择。"""
    opts = {
        "diameter": {"method": "raw", "value": 10.0},
        "curvature": {"method": "raw", "value": 0.01},
        "kappa": {"method": "raw", "value": 0.0},
        "alpha4": {"method": "raw", "value": 1e-4},
        "alpha6": {"method": "raw", "value": 1e-6},
    }
    a_new = Asphere.from_options(POPULATION, opts)
    a_old = a_new.clone()

    idx = torch.arange(POPULATION, dtype=torch.long)
    a_new.mutate_(idx, {"alpha": 1e-5, "curvature": 1e-3, "kappa": 0.05})

    mask = _mask(a_new.device)
    merged = Asphere.where(mask, a_new, a_old)

    for name in ("Diameter", "Curvature", "Kappa", "Alpha", "Mask"):
        expected = _expected_where(mask, getattr(a_new, name), getattr(a_old, name))
        assert torch.equal(getattr(merged, name), expected), (
            f"asphere where wrong for {name}"
        )


# ═══════════════════════════════════════════════════════════════════════
# runner
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # Collect all test_* functions in this module
    mod = sys.modules[__name__]
    tests = [(n, fn) for n, fn in vars(mod).items() if n.startswith("test_") and callable(fn)]

    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:
            print(f"  FAIL  {name}")
            import traceback

            traceback.print_exc()

    print(f"\n{passed}/{len(tests)} passed")
    if passed < len(tests):
        sys.exit(1)
