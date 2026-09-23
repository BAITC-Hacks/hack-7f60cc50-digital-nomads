"""Загрузка данных и проверка консистентности (логика стартового кода организаторов + отчёт)."""
from pathlib import Path

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
        return pd.read_csv(csv)
    raise FileNotFoundError(f"нет {pq} (или {csv})")


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
    for df, cols in [(edges, ["src", "dst"]), (nodes, ["gid"]), (tx, ["src", "dst"])]:
        for c in cols:
            df[c] = df[c].astype("int64")
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
    rep["unknown_gids_in_edges"] = int(len(in_edges - set(nodes.gid)))

    if verbose:
        print("=" * 64)
        print("ПРОВЕРКА ДАННЫХ")
        print("=" * 64)
        for k, v in rep.items():
            print(f"  {k:<24}: {v:,}" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"  {k:<24}: {v}")
        print("=" * 64)
    return rep
