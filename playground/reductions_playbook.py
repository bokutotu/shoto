"""ISL/islpy playbook: dependence analysis + scheduling (incl. reductions).

Run:
    cd shoto/playground
    uv run python reductions_playbook.py

Notes:
- Schedule *legality* should be checked against (over-approximated) dependences.
  In isl's flow API, that means using *may* dependences, not *must*.
- If you want to parallelize/transform a reduction update, you typically need to
  *remove* reduction-carried dependences from validity and handle them in
  lowering (privatization/atomics/tree-reduce). This file only models the
  scheduling side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import islpy as isl


@dataclass(frozen=True)
class ReductionSpec:
    """A subset of statement instances that perform a reduction update."""

    domain: isl.UnionSet
    write: isl.UnionMap
    read: isl.UnionMap

    def is_empty(self) -> bool:
        return self.domain.is_empty() or self.write.is_empty()


@dataclass(frozen=True)
class KernelModel:
    name: str
    context: isl.Set
    domain: isl.UnionSet
    program_order: isl.UnionMap
    reads: isl.UnionMap
    writes: isl.UnionMap
    reduction: ReductionSpec | None


@dataclass(frozen=True)
class Dependences:
    raw: isl.UnionMap
    reduction_carried: isl.UnionMap
    validity_no_reduction: isl.UnionMap


def _empty_umap() -> isl.UnionMap:
    return isl.UnionMap("{ }")


def _bounded(umap: isl.UnionMap, domain: isl.UnionSet) -> isl.UnionMap:
    # Important: keep maps bounded by the domain.
    # If left unconstrained (e.g. "S[i,j] -> ..." with no bounds), isl's
    # dependence analysis may fail ("unbounded optimum").
    return umap.intersect_domain(domain)


def _union_maps(maps: Iterable[isl.UnionMap]) -> isl.UnionMap:
    out = _empty_umap()
    for m in maps:
        out = out.union(m)
    return out


def _may_dependence(*, sources: isl.UnionMap, sinks: isl.UnionMap, order: isl.UnionMap) -> isl.UnionMap:
    """Compute a conservative (may) dependence relation: sources -> sinks."""

    if sources.is_empty() or sinks.is_empty() or order.is_empty():
        return _empty_umap()

    ai = isl.UnionAccessInfo.from_sink(sinks)
    ai = ai.set_may_source(sources)
    ai = ai.set_schedule_map(order)
    return ai.compute_flow().get_may_dependence()


def _reduction_carried_deps(reduction: ReductionSpec | None, *, order: isl.UnionMap) -> isl.UnionMap:
    """Dependences internal to the reduction update (accumulator-carried)."""

    if reduction is None or reduction.is_empty():
        return _empty_umap()

    deps = _may_dependence(sources=reduction.write, sinks=reduction.write, order=order)

    # If the update also reads the accumulator, include true and anti dependences.
    # (We model statement instances as atomic; this is still the right "carried"
    # set to drop from validity when you plan to lower the reduction specially.)
    if not reduction.read.is_empty():
        deps = deps.union(_may_dependence(sources=reduction.write, sinks=reduction.read, order=order))
        deps = deps.union(_may_dependence(sources=reduction.read, sinks=reduction.write, order=order))

    return deps


def analyze_dependences(model: KernelModel) -> Dependences:
    raw = _may_dependence(sources=model.writes, sinks=model.reads, order=model.program_order)
    waw = _may_dependence(sources=model.writes, sinks=model.writes, order=model.program_order)
    war = _may_dependence(sources=model.reads, sinks=model.writes, order=model.program_order)
    all_deps = raw.union(waw).union(war)

    red = _reduction_carried_deps(model.reduction, order=model.program_order)
    validity = all_deps.subtract(red)

    return Dependences(
        raw=raw,
        reduction_carried=red,
        validity_no_reduction=validity,
    )


def compute_schedule(*, domain: isl.UnionSet, validity: isl.UnionMap, coincidence: isl.UnionMap, proximity: isl.UnionMap) -> isl.Schedule:
    """Compute a schedule with explicit validity/coincidence/proximity constraints."""

    sc = isl.ScheduleConstraints.on_domain(domain)
    sc = sc.set_validity(validity)
    sc = sc.set_coincidence(coincidence)
    sc = sc.set_proximity(proximity)
    return sc.compute_schedule()


def _ast_to_c_str(*, schedule: isl.Schedule, context: isl.Set) -> str:
    build = isl.AstBuild.from_context(context)
    return build.node_from_schedule(schedule).to_C_str()


def transform_schedule_reorder(schedule: isl.Schedule, order: list[int]) -> isl.Schedule:
    """
    指定されたインデックスのリスト順にループの次元を並び替えます。
    バンドの次元数とリストの長さが一致する場合に適用されます。
    
    Args:
        schedule: 対象のisl.Schedule
        order: 新しい次元順序を指定する整数のリスト
               例: 3次元バンドに対して [2, 0, 1] を指定すると (i, j, k) -> (k, i, j) に変換
    """
    def _reorder_callback(node):
        if node.get_type() != isl.schedule_node_type.band:
            return node

        n_members = node.band_n_member()

        if len(order) != n_members:
            return node

        if not set(order).issubset(set(range(n_members))):
            return node

        partial_sched = node.band_get_partial_schedule()

        space = node.band_get_space() 
        range_space = space.range()
        map_space = range_space.map_from_set() # { Range -> Range }

        old_ma = isl.MultiAff.identity(map_space)
        new_ma = isl.MultiAff.identity(map_space)

        for i, source_idx in enumerate(order):
            aff = old_ma.get_aff(source_idx)
            new_ma = new_ma.set_aff(i, aff)

        new_partial_sched = partial_sched.apply_multi_aff(new_ma)
        child_node = node.delete()
        new_node = child_node.insert_partial_schedule(new_partial_sched)
        return new_node

    return schedule.map_schedule_node_bottom_up(_reorder_callback)


def build_kernel_model(
    *,
    name: str,
    params: str,
    domain: str,
    program_order: str,
    writes: str,
    reads: Sequence[str],
    reduction_domain: str | None = None,
    reduction_write: str | None = None,
    reduction_read: str | None = None,
) -> KernelModel:
    dom = isl.UnionSet(domain)
    order = _bounded(isl.UnionMap(program_order), dom)

    wr = _bounded(isl.UnionMap(writes), dom)
    rd = _union_maps(_bounded(isl.UnionMap(r), dom) for r in reads)
    rd = rd.intersect_domain(dom)

    ctx = isl.Set(params)

    reduction: ReductionSpec | None = None
    if reduction_domain is not None and reduction_write is not None:
        red_dom = isl.UnionSet(reduction_domain).intersect(dom)
        red_wr = _bounded(isl.UnionMap(reduction_write), red_dom)
        red_rd = _empty_umap()
        if reduction_read is not None:
            red_rd = _bounded(isl.UnionMap(reduction_read), red_dom)
        reduction = ReductionSpec(domain=red_dom, write=red_wr, read=red_rd)

    return KernelModel(
        name=name,
        context=ctx,
        domain=dom,
        program_order=order,
        reads=rd,
        writes=wr,
        reduction=reduction,
    )


def demo_kernel(model: KernelModel) -> None:
    print("=" * 80)
    print(model.name)
    print("=" * 80)

    deps = analyze_dependences(model)

    # Reduction-relaxed schedule:
    # - legality uses conservative dependences with reduction-carried deps removed
    # - proximity still includes reduction-carried deps to encourage locality
    schedule = compute_schedule(
        domain=model.domain,
        validity=deps.validity_no_reduction,
        coincidence=deps.validity_no_reduction,
        proximity=deps.raw.union(deps.reduction_carried),
    )

    schedule = transform_schedule_reorder(schedule, [2, 1, 0])

    print(_ast_to_c_str(schedule=schedule, context=model.context))


def main() -> None:
    # ---------------------------------------------------------------------
    # Fused GEMM + Add + ReLU (single kernel shape)
    #
    #   for i,j:
    #     acc = 0
    #     for k:
    #       acc += A[i,k] * B[k,j]        (reduction over k)
    #     Y[i,j] = relu(acc + Bias[i,j])
    #
    # We model this as 3 statements:
    # - S_init[i,j]   : initialize accumulator C[i,j]
    # - S_gemm[i,j,k] : reduction update on C[i,j]
    # - S_out[i,j]    : read C[i,j] and Bias, write Y[i,j]
    #
    # This shows how "fusion" is expressed: multiple statements scheduled
    # into one loop nest (same i,j loops), with ordering constraints.
    # ---------------------------------------------------------------------
    model = build_kernel_model(
        name="fused gemm + add + relu (single kernel)",
        params="[N,M,K] -> { : 0 < K}",
        domain=(
            "[N,M,K] -> { "
            "S_init[i,j] : 0 <= i < N and 0 <= j < M; "
            "S_gemm[i,j,k] : 0 <= i < N and 0 <= j < M and 0 <= k < K; "
            "S_out[i,j] : 0 <= i < N and 0 <= j < M "
            "}"
        ),
        program_order=(
            "[N,M,K] -> { "
            "S_init[i,j] -> [i,j,0]; "
            "S_gemm[i,j,k] -> [i,j,k+1]; "
            "S_out[i,j] -> [i,j,K+1] "
            "}"
        ),
        # Writes: init and gemm update C; out writes Y
        writes=(
            "[N,M,K] -> { "
            "S_init[i,j] -> C[i,j]; "
            "S_gemm[i,j,k] -> C[i,j]; "
            "S_out[i,j] -> Y[i,j] "
            "}"
        ),
        # Reads: gemm reads A,B and (accumulator) C; out reads C and Bias
        reads=[
            "[N,M,K] -> { S_gemm[i,j,k] -> A[i,k] }",
            "[N,M,K] -> { S_gemm[i,j,k] -> B[k,j] }",
            "[N,M,K] -> { S_gemm[i,j,k] -> C[i,j] }",
            "[N,M,K] -> { S_out[i,j] -> C[i,j] }",
            "[N,M,K] -> { S_out[i,j] -> Bias[i,j] }",
        ],
        # Reduction is the update statement instances on accumulator C[i,j].
        reduction_domain="[N,M,K] -> { S_gemm[i,j,k] }",
        reduction_write="[N,M,K] -> { S_gemm[i,j,k] -> C[i,j] }",
        reduction_read="[N,M,K] -> { S_gemm[i,j,k] -> C[i,j] }",
    )
    demo_kernel(model)


if __name__ == "__main__":
    main()
