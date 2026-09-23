"""Загрузка данных и проверка консистентности (логика стартового кода организаторов + отчёт)."""
from pathlib import Path

import networkx as nx
import pandas as pd

REQUIRED = {
    "edges": {"src", "dst", "sum_kzt", "n_tx", "depth"},
    "nodes": {"gid", "depth", "is_seed"},
    "transactions": {"src", "dst", "date", "sum_kzt"},
}


def _read(data_dir: Path, name: str) -> pd.DataFrame:
    pq, csv = data_dir / f"{name}.parquet", data_dir / f"{name}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():                       # запасной вариант для синтетики / машин без pyarrow
        # gid читаем как int64 сразу: 18-значные числа не должны пройти через float
        ids = {c: "int64" for c in ("gid", "src", "dst") if c in pd.read_csv(csv, nrows=0).columns}
        return pd.read_csv(csv, dtype=ids)
    raise FileNotFoundError(f"нет {pq} (или {csv}). Положите файлы датасета в папку {data_dir}/ "
                            f"(edges, nodes, transactions) или проверьте на синтетике: make demo")


def _as_gid(df: pd.DataFrame, name: str, c: str) -> pd.Series:
    """18-значные gid не помещаются в float64 (точность 2^53 ≈ 9·10^15): при float-колонке
    соседние gid слипаются. Поэтому float и пропуски — ошибка с объяснением, а не тихое округление."""
    s = df[c]
    if s.isna().any():
        raise ValueError(f"{name}.{c}: {int(s.isna().sum())} пустых gid")
    if s.dtype.kind == "f":
        raise ValueError(f"{name}.{c}: gid пришёл как float — 18-значные номера уже искажены. "
                         f"Сохраните колонку как int64 или строку")
    return s.astype("int64")


def load(data_dir):
    data_dir = Path(data_dir)
    out = {}
    for name, cols in REQUIRED.items():
        df = _read(data_dir, name)
        miss = cols - set(df.columns)
        if miss:
            raise ValueError(f"{name}: нет колонок {miss}. Сверьтесь с README датасета.")
        out[name] = df
    edges, nodes, tx = out["edges"], out["nodes"], out["transactions"]
    for name, df, cols in [("edges", edges, ["src", "dst"]), ("nodes", nodes, ["gid"]),
                           ("transactions", tx, ["src", "dst"])]:
        for c in cols:
            df[c] = _as_gid(df, name, c)
    if nodes.gid.duplicated().any():
        raise ValueError(f"nodes: {int(nodes.gid.duplicated().sum())} повторяющихся gid")
    nodes["is_seed"] = nodes["is_seed"].astype(bool)
    nodes["depth"] = nodes["depth"].astype(int)
    edges["sum_kzt"] = edges["sum_kzt"].astype(float)
    tx["sum_kzt"] = tx["sum_kzt"].astype(float)
    tx["date"] = pd.to_datetime(tx["date"]).dt.normalize()
    return edges, nodes, tx


def sanity_check(edges, nodes, tx, verbose=True):
    """Возвращает словарь-отчёт о качестве данных. Ничего не «чинит» молча."""
    rep = {
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
        "n_tx": int(len(tx)),
        "n_seed": int(nodes.is_seed.sum()),
        "turnover_kzt": float(edges.sum_kzt.sum()),
        "period": f"{tx.date.min().date()} — {tx.date.max().date()}",
        "depth_counts": {int(k): int(v) for k, v in nodes.depth.value_counts().sort_index().items()},
    }
    agg = tx.groupby(["src", "dst"]).agg(s=("sum_kzt", "sum"), c=("sum_kzt", "size")).reset_index()
    m = edges.merge(agg, on=["src", "dst"], how="outer", indicator=True)
    rep["edges_match_tx_pairs"] = bool((m._merge == "both").all())
    both = m[m._merge == "both"]
    rep["edges_match_tx_sums"] = bool(((both.sum_kzt - both.s).abs() < 1).all())
    rep["edges_match_tx_counts"] = bool((both.n_tx == both.c).all())

    in_edges = set(edges.src) | set(edges.dst)
    seeds = set(nodes.loc[nodes.is_seed, "gid"])
    rep["orphans"] = int(len(set(nodes.gid) - in_edges))
    rep["seed_not_in_edges"] = int(len(seeds - in_edges))
    rep["seed_only_receivers"] = int(len((seeds & set(edges.dst)) - set(edges.src)))
    rep["seed_without_outgoing"] = int(len(seeds - set(edges.src)))
    rep["depth4_zero_out"] = int(((nodes.depth == 4) & ~nodes.gid.isin(set(edges.src))).sum())
    rep["tx_below_5000"] = int((tx.sum_kzt < 5000).sum())
    rep["tx_min_kzt"] = float(tx.sum_kzt.min()) if len(tx) else None
    rep["unknown_gids_in_edges"] = int(len(in_edges - set(nodes.gid)))

    # Неполнота входящих — не ошибка данных, а свойство 4-hop-выгрузки. Считаем обе
    # трактовки явно: 354 — узлы с наблюдаемым входом и out/in > 1 из ТЗ; ещё 23
    # отправителя имеют нулевой видимый вход и потому не получают конечный pass_through.
    incoming = edges.groupby("dst").sum_kzt.sum()
    outgoing = edges.groupby("src").sum_kzt.sum()
    flow = pd.DataFrame({"incoming": incoming, "outgoing": outgoing}).fillna(0.0)
    has_visible_in = flow.incoming > 0
    pass_through = flow.outgoing / flow.incoming.where(has_visible_in)
    rep["pass_through_gt_1"] = int((pass_through > 1).sum())
    rep["pass_through_0_8_to_1_2"] = int(pass_through.between(0.8, 1.2).sum())
    rep["out_gt_visible_in_including_zero_in"] = int((flow.outgoing > flow.incoming).sum())
    rep["outgoing_with_zero_visible_in"] = int(((flow.outgoing > 0) & ~has_visible_in).sum())

    # Число 16 в ТЗ относится к графу рёбер: 19 seed, отсутствующие в рёбрах,
    # рассматриваются отдельно. Если добавить их как изолированные вершины, компонент 35.
    G = nx.DiGraph()
    G.add_edges_from((int(r.src), int(r.dst)) for r in edges.itertuples(index=False))
    components = sorted((len(c) for c in nx.weakly_connected_components(G)), reverse=True)
    rep["weak_components_with_edges"] = int(len(components))
    rep["weak_component_sizes"] = components
    rep["weak_components_including_isolates"] = int(len(components) + rep["orphans"])

    # Контрольная «базовая Louvain» из примечания ТЗ: неориентированная проекция,
    # сырой денежный вес, seed=42. Рабочая кластеризация намеренно использует log1p-вес
    # и отдельно группирует изоляты, поэтому её число сообществ может отличаться.
    UG = nx.Graph()
    for r in edges.itertuples(index=False):
        u, v, amount = int(r.src), int(r.dst), float(r.sum_kzt)
        if UG.has_edge(u, v):
            UG[u][v]["sum_kzt"] += amount
        else:
            UG.add_edge(u, v, sum_kzt=amount)
    communities = nx.community.louvain_communities(UG, weight="sum_kzt", resolution=1.0, seed=42)
    rep["baseline_louvain_communities"] = int(len(communities))
    rep["baseline_louvain_multi_seed"] = int(sum(len(c & seeds) > 1 for c in communities))

    if verbose:
        print("=" * 64)
        print("ПРОВЕРКА ДАННЫХ")
        print("=" * 64)
        for k, v in rep.items():
            print(f"  {k:<24}: {v:,}" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"  {k:<24}: {v}")
        print("=" * 64)
    return rep
