import islpy as isl


def compute_deps(src, sink, schedule):
    access_info = isl.UnionAccessInfo.from_sink(sink)
    access_info = access_info.set_may_source(src)
    access_info = access_info.set_schedule_map(schedule)
    flow = access_info.compute_flow()
    return flow.get_may_dependence()


def main():
    domain = isl.UnionSet("[N] -> { S[i, j, k] : 0 <= i < N and 0 <= j < N and 0 <= k < N }")

    schedule = isl.UnionMap("[N] -> { S[i, j, k] -> [i, j, k] }")

    write = isl.UnionMap("[N] -> { S[i, j, k] -> C[i, j] }")

    read_C = isl.UnionMap("[N] -> { S[i, j, k] -> C[i, j] }")
    read_A = isl.UnionMap("[N] -> { S[i, j, k] -> A[i, k] }")
    read_B = isl.UnionMap("[N] -> { S[i, j, k] -> B[k, j] }")

    read = read_C.union(read_A).union(read_B)

    raw_deps = compute_deps(write, read, schedule)
    waw_deps = compute_deps(write, write, schedule)
    war_deps = compute_deps(read, write, schedule)

    valid_deps = raw_deps.union(waw_deps).union(war_deps)

    sc = isl.ScheduleConstraints.on_domain(domain)
    sc = sc.set_validity(valid_deps)
    # 計算の正しさには関係ないが、高速化（キャッシュ効率）のために『なるべく時間的に近く』実行してほしい
    sc = sc.set_proximity(valid_deps)

    new_schedule = sc.compute_schedule()

    print("Computed Schedule:")
    print(new_schedule)

    build = isl.AstBuild.from_context(isl.Set("[N] -> { : }"))
    ast_node = build.node_from_schedule(new_schedule)
    print("Generated AST:")
    print(ast_node)


if __name__ == "__main__":
    main()
