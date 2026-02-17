"""
ISLのスケジュール合成の動作を確認するテスト

Set/Mapを個別に作成してから組み合わせる
パラメータ（N, M）を使用
"""

import islpy as isl


def test_separate_domains_and_schedules():
    """
    別々にドメインとスケジュールマップを作成して融合する
    """
    ctx = isl.Context()

    print("=== 1. 個別にドメインを作成 ===")

    # 個別にドメインを作成（パラメータ N, M を使用）
    domain_s = isl.UnionSet.read_from_str(ctx, "[N,M] -> { S[i,j] : 0 <= i < N and 0 <= j < M }")
    domain_k = isl.UnionSet.read_from_str(ctx, "[N,M] -> { K[x,y] : 0 <= x < N and 0 <= y < M }")

    print(f"S のドメイン: {domain_s}")
    print(f"K のドメイン: {domain_k}")
    print()

    print("=== 2. 個別にスケジュールマップを作成 ===")

    # 個別にスケジュールマップを作成
    # S[i,j] -> [j, i] (軸交換)
    sched_map_s = isl.UnionMap.read_from_str(ctx, "[N,M] -> { S[i,j] -> [j, i] }")
    # K[x,y] -> [x, y] (デフォルト順)
    sched_map_k = isl.UnionMap.read_from_str(ctx, "[N,M] -> { K[x,y] -> [x, y] }")

    print(f"S のスケジュールマップ: {sched_map_s}")
    print(f"K のスケジュールマップ: {sched_map_k}")
    print()

    print("=== 3. ドメインとスケジュールマップを融合 ===")

    # ドメインを融合
    combined_domain = domain_s.union(domain_k)
    print(f"融合後のドメイン: {combined_domain}")

    # スケジュールマップを融合
    combined_sched_map = sched_map_s.union(sched_map_k)
    print(f"融合後のスケジュールマップ: {combined_sched_map}")
    print()

    print("=== 4. スケジュールを構築 ===")

    # ドメインからスケジュールを作成
    schedule = isl.Schedule.from_domain(combined_domain)
    print(f"ドメインのみのスケジュール: {schedule}")

    # ルートノードを取得してスケジュールマップを挿入
    root = schedule.get_root()

    # 子ノード（ドメインの下）に移動
    node = root.child(0)

    # バンドノードを挿入（スケジュールマップを使用）
    mupa = isl.MultiUnionPwAff.from_union_map(combined_sched_map)
    node = node.insert_partial_schedule(mupa)

    # スケジュールを取得
    schedule = node.get_schedule()
    print(f"最終スケジュール: {schedule}")
    print()

    print("=== 5. AST生成 ===")

    # パラメータ N, M に制約を付ける（正の値）
    context = isl.Set.read_from_str(ctx, "[N,M] -> { : N > 0 and M > 0 }")
    build = isl.AstBuild.from_context(context)
    ast = build.node_from_schedule(schedule)

    printer = isl.Printer.to_str(ctx)
    printer = printer.set_output_format(isl.format.C)
    printer = printer.print_ast_node(ast)
    print("生成されたコード:")
    print(printer.get_str())
    print()


def test_schedule_from_constraints():
    """
    依存関係から自動的にスケジュールを計算させる
    """
    ctx = isl.Context()

    print("=== 依存関係からスケジュールを自動計算 ===")

    # ドメイン（パラメータ使用）
    domain = isl.UnionSet.read_from_str(
        ctx,
        "[N,M] -> { S[i,j] : 0 <= i < N and 0 <= j < M; K[x,y] : 0 <= x < N and 0 <= y < M }"
    )
    print(f"ドメイン: {domain}")

    # 依存関係: S[i,j] の後に K[i,j] を実行（同じインデックスで）
    validity = isl.UnionMap.read_from_str(ctx, "[N,M] -> { S[i,j] -> K[i,j] }")
    print(f"依存関係 (validity): {validity}")
    print()

    # スケジュール制約を構築
    sc = isl.ScheduleConstraints.on_domain(domain)
    sc = sc.set_validity(validity)

    # ISLに最適なスケジュールを計算させる
    schedule = sc.compute_schedule()
    print(f"自動計算されたスケジュール: {schedule}")
    print()

    # AST生成
    context = isl.Set.read_from_str(ctx, "[N,M] -> { : N > 0 and M > 0 }")
    build = isl.AstBuild.from_context(context)
    ast = build.node_from_schedule(schedule)

    printer = isl.Printer.to_str(ctx)
    printer = printer.set_output_format(isl.format.C)
    printer = printer.print_ast_node(ast)
    print("生成されたコード:")
    print(printer.get_str())
    print()


def test_two_independent_schedules():
    """
    2つの独立したスケジュールを別々に作成して融合
    """
    ctx = isl.Context()

    print("=== 2つの独立したスケジュールを融合 ===")

    # スケジュール1: S用
    domain_s = isl.UnionSet.read_from_str(ctx, "[N,M] -> { S[i,j] : 0 <= i < N and 0 <= j < M }")
    sched_s = isl.Schedule.from_domain(domain_s)
    root_s = sched_s.get_root()
    node_s = root_s.child(0)
    mupa_s = isl.MultiUnionPwAff.from_union_map(
        isl.UnionMap.read_from_str(ctx, "[N,M] -> { S[i,j] -> [j, i] }")
    )
    node_s = node_s.insert_partial_schedule(mupa_s)
    sched_s = node_s.get_schedule()

    print(f"スケジュールS: {sched_s}")

    # スケジュール2: K用
    domain_k = isl.UnionSet.read_from_str(ctx, "[N,M] -> { K[x,y] : 0 <= x < N and 0 <= y < M }")
    sched_k = isl.Schedule.from_domain(domain_k)
    root_k = sched_k.get_root()
    node_k = root_k.child(0)
    mupa_k = isl.MultiUnionPwAff.from_union_map(
        isl.UnionMap.read_from_str(ctx, "[N,M] -> { K[x,y] -> [x, y] }")
    )
    node_k = node_k.insert_partial_schedule(mupa_k)
    sched_k = node_k.get_schedule()

    print(f"スケジュールK: {sched_k}")
    print()

    # 2つのスケジュールのドメインとマップを取り出して手動で融合
    print("=== 手動でドメインとマップを融合 ===")

    # ドメインを融合
    combined_domain = domain_s.union(domain_k)

    # スケジュールマップを融合
    sched_map_s = isl.UnionMap.read_from_str(ctx, "[N,M] -> { S[i,j] -> [j, i] }")
    sched_map_k = isl.UnionMap.read_from_str(ctx, "[N,M] -> { K[x,y] -> [x, y] }")
    combined_map = sched_map_s.union(sched_map_k)

    # 新しいスケジュールを構築
    combined_sched = isl.Schedule.from_domain(combined_domain)
    root = combined_sched.get_root()
    node = root.child(0)
    mupa = isl.MultiUnionPwAff.from_union_map(combined_map)
    node = node.insert_partial_schedule(mupa)
    combined_sched = node.get_schedule()

    print(f"融合後のスケジュール: {combined_sched}")
    print()

    # AST生成
    context = isl.Set.read_from_str(ctx, "[N,M] -> { : N > 0 and M > 0 }")
    build = isl.AstBuild.from_context(context)
    ast = build.node_from_schedule(combined_sched)

    printer = isl.Printer.to_str(ctx)
    printer = printer.set_output_format(isl.format.C)
    printer = printer.print_ast_node(ast)
    print("生成されたコード:")
    print(printer.get_str())
    print()


def test_sequence_vs_set():
    """
    Sequence（順序あり）vs Set（順序なし）の違い
    APIを使って構築する
    """
    ctx = isl.Context()

    print("=== Sequence vs Set の違い ===")

    # ドメイン
    domain = isl.UnionSet.read_from_str(ctx, "[N] -> { S[i] : 0 <= i < N; K[i] : 0 <= i < N }")

    # 依存関係を設定して Sequence を自動生成
    # S[i] -> K[i] の依存（同じiでSの後にKを実行）
    validity = isl.UnionMap.read_from_str(ctx, "[N] -> { S[i] -> K[i] }")

    sc = isl.ScheduleConstraints.on_domain(domain)
    sc = sc.set_validity(validity)
    sched_seq = sc.compute_schedule()

    print("Sequence (依存関係 S[i] -> K[i] あり):")
    print(f"スケジュール: {sched_seq}")
    context = isl.Set.read_from_str(ctx, "[N] -> { : N > 0 }")
    build = isl.AstBuild.from_context(context)
    ast = build.node_from_schedule(sched_seq)
    printer = isl.Printer.to_str(ctx)
    printer = printer.set_output_format(isl.format.C)
    printer = printer.print_ast_node(ast)
    print("生成されたコード:")
    print(printer.get_str())
    print()

    # 依存関係なしで同じスケジュールマップを使う（融合される）
    sched_map = isl.UnionMap.read_from_str(ctx, "[N] -> { S[i] -> [i]; K[i] -> [i] }")
    sched_fused = isl.Schedule.from_domain(domain)
    root = sched_fused.get_root()
    node = root.child(0)
    mupa = isl.MultiUnionPwAff.from_union_map(sched_map)
    node = node.insert_partial_schedule(mupa)
    sched_fused = node.get_schedule()

    print("融合 (依存関係なし、同じ時間座標):")
    print(f"スケジュール: {sched_fused}")
    ast = build.node_from_schedule(sched_fused)
    printer = isl.Printer.to_str(ctx)
    printer = printer.set_output_format(isl.format.C)
    printer = printer.print_ast_node(ast)
    print("生成されたコード:")
    print(printer.get_str())
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("テスト1: 個別にドメインとスケジュールマップを作成して融合")
    print("=" * 60)
    test_separate_domains_and_schedules()

    print("=" * 60)
    print("テスト2: 依存関係からスケジュールを自動計算")
    print("=" * 60)
    test_schedule_from_constraints()

    print("=" * 60)
    print("テスト3: 2つの独立したスケジュールを融合")
    print("=" * 60)
    test_two_independent_schedules()

    print("=" * 60)
    print("テスト4: Sequence vs Set の違い")
    print("=" * 60)
    test_sequence_vs_set()
