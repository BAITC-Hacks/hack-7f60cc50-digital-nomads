"""
Метрики узлов. Все признаки интерпретируемы: каждую можно произнести вслух жюри.

Структура:     in/out степени, обороты, число переводов, pass_through, PageRank, HITS, betweenness
Происхождение: seed_reach (сколько разных seed «дотягиваются» до узла), merge_gain (сколько
               seed-потоков узел впервые сводит вместе), seed_traced_in_kzt (сколько входящих
               денег прослеживается до seed по цепочке)
Время:         лаг между поступлением и отправкой, доля «сквозного» пропуска, синхронные плательщики
Аномалии:      дробление у порога, повторяющиеся суммы, робастный z-score внутри колена
"""
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd


def build_graph(edges: pd.DataFrame, nodes: pd.DataFrame) -> nx.DiGraph:
    """Направленный взвешенный граф. Узлы без рёбер тоже добавляются — они обязаны попасть в выгрузку."""
    G = nx.DiGraph()
    G.add_nodes_from(int(g) for g in nodes.gid)
    for r in edges.itertuples(index=False):
        G.add_edge(int(r.src), int(r.dst), sum_kzt=float(r.sum_kzt), n_tx=int(r.n_tx), depth=int(r.depth))
    return G


def pct_pos(s: pd.Series) -> pd.Series:
    """Перцентиль среди положительных значений; нули остаются нулями (иначе 60% нулей получат 0.3)."""
    s = s.fillna(0).astype(float)
    out = pd.Series(0.0, index=s.index)
    pos = s > 0
    if pos.any():
        out[pos] = s[pos].rank(pct=True, method="average")
    return out


def structural(G: nx.DiGraph, nodes: pd.DataFrame) -> pd.DataFrame:
    df = nodes[["gid", "depth", "is_seed"]].copy().reset_index(drop=True)
    g = df.gid
    df["in_deg"] = g.map(dict(G.in_degree())).fillna(0).astype(int)
    df["out_deg"] = g.map(dict(G.out_degree())).fillna(0).astype(int)
    df["in_kzt"] = g.map(dict(G.in_degree(weight="sum_kzt"))).fillna(0.0)
    df["out_kzt"] = g.map(dict(G.out_degree(weight="sum_kzt"))).fillna(0.0)
    df["in_tx"] = g.map(dict(G.in_degree(weight="n_tx"))).fillna(0).astype(int)
    df["out_tx"] = g.map(dict(G.out_degree(weight="n_tx"))).fillna(0).astype(int)
    df["pass_through"] = np.where(df.in_kzt > 0, df.out_kzt / df.in_kzt.where(df.in_kzt > 0), np.nan)
    df["fan_ratio"] = df.out_deg / df.in_deg.clip(lower=1)

    df["pagerank"] = g.map(nx.pagerank(G, weight="sum_kzt")).fillna(0.0)
    try:
        hubs, auth = nx.hits(G, max_iter=2000, normalized=True)
    except nx.PowerIterationFailedConvergence:
        hubs, auth = {}, {}
    df["hub"] = g.map(hubs).fillna(0.0)
    df["authority"] = g.map(auth).fillna(0.0)
    # betweenness без весов: «через скольких посредников проходят кратчайшие денежные маршруты»
    df["betweenness"] = g.map(nx.betweenness_centrality(G, normalized=True)).fillna(0.0)
    df["truncated_by_depth"] = (df.depth == 4) & (df.out_deg == 0)
    df["component"] = 0
    for i, comp in enumerate(sorted(nx.weakly_connected_components(G), key=len, reverse=True)):
        df.loc[df.gid.isin(comp), "component"] = i
    return df


def provenance(G: nx.DiGraph, df: pd.DataFrame, max_iter: int = 30) -> pd.DataFrame:
    """Откуда пришли деньги — ключевой блок кейса."""
    seeds = set(df.loc[df.is_seed, "gid"])
    reach = defaultdict(set)                      # gid -> множество seed, из которых есть путь
    for s in seeds:
        for v in nx.descendants(G, s):
            reach[v].add(s)
    df["seed_reach"] = df.gid.map(lambda v: len(reach.get(v, ()))).astype(int)

    # merge_gain: сколько seed-потоков узел сводит вместе сверх самого «богатого» плательщика.
    # Дроп распределителя наследует охват плательщика → 0. Консолидатор 6 seed → 6.
    def reach_incl(u):
        return len(reach.get(u, ())) + (1 if u in seeds else 0)
    gain, agg = {}, {}
    for v in G.nodes:
        preds = list(G.predecessors(v))
        if not preds:
            gain[v], agg[v] = 0, 0
            continue
        gain[v] = max(0, len(reach.get(v, ())) - max(reach_incl(u) for u in preds))
        # плательщики, которые сами уже несут деньги ≥2 seed — «консолидация второго уровня»
        agg[v] = sum(1 for u in preds if reach_incl(u) >= 2)
    df["merge_gain"] = df.gid.map(gain).fillna(0).astype(int)
    df["n_agg_payers"] = df.gid.map(agg).fillna(0).astype(int)

    # прямые плательщики-seed
    n_sp, sp_kzt = {}, {}
    for v in G.nodes:
        ps = [u for u in G.predecessors(v) if u in seeds]
        n_sp[v] = len(ps)
        sp_kzt[v] = sum(G[u][v]["sum_kzt"] for u in ps)
    df["n_seed_payers"] = df.gid.map(n_sp).fillna(0).astype(int)
    df["seed_in_kzt"] = df.gid.map(sp_kzt).fillna(0.0)

    # seed_traced_in_kzt: пропорциональная трассировка. Доля исходящих узла, «объяснимая»
    # деньгами из seed-цепочки: f(seed)=1; f(v)=min(1, traced_in(v)/out_kzt(v)).
    # Если узел отдал больше, чем получил в графе, излишек пришёл извне → f < 1.
    out_kzt = dict(G.out_degree(weight="sum_kzt"))
    f = {v: (1.0 if v in seeds else 0.0) for v in G.nodes}
    traced = {v: 0.0 for v in G.nodes}
    for _ in range(max_iter):
        new_traced = {v: 0.0 for v in G.nodes}
        for u, v, d in G.edges(data=True):
            new_traced[v] += d["sum_kzt"] * f[u]
        delta = 0.0
        for v in G.nodes:
            if v in seeds:
                continue
            nf = min(1.0, new_traced[v] / out_kzt[v]) if out_kzt[v] > 0 else 1.0
            delta += abs(nf - f[v])
            f[v] = nf
        traced = new_traced
        if delta < 1e-9:
            break
    df["seed_traced_in_kzt"] = df.gid.map(traced).fillna(0.0)
    return df


def temporal(tx: pd.DataFrame, df: pd.DataFrame, fast_days: int = 2) -> pd.DataFrame:
    """Временные паттерны по отдельным транзакциям."""
    tx = tx.copy()
    tx["day"] = (tx.date - tx.date.min()).dt.days
    last_day = int(tx.day.max()) if len(tx) else 30
    inc = tx.rename(columns={"dst": "node"})[["node", "src", "day", "sum_kzt", "date"]]
    out = tx.rename(columns={"src": "node"})[["node", "dst", "day", "sum_kzt", "date"]]

    g_in = inc.groupby("node")
    g_out = out.groupby("node")
    feats = pd.DataFrame(index=df.gid)
    feats["first_in_day"] = g_in.day.min()
    feats["last_in_day"] = g_in.day.max()
    feats["first_out_day"] = g_out.day.min()
    feats["active_in_days"] = g_in.day.nunique()
    feats["mean_in_ticket"] = g_in.sum_kzt.mean()

    # лаг: для каждого исходящего — сколько дней прошло с последнего поступления
    a = out.sort_values("day")[["node", "day", "sum_kzt"]].rename(columns={"sum_kzt": "out_amt"})
    b = inc.sort_values("day")[["node", "day"]].rename(columns={"day": "in_day"})
    b["day"] = b["in_day"]
    m = pd.merge_asof(a, b, on="day", by="node", direction="backward")
    m["lag"] = m.day - m.in_day
    has_in = m.in_day.notna()
    fast = m[has_in & (m.lag <= fast_days)].groupby("node").out_amt.sum()
    tot = m.groupby("node").out_amt.sum()
    feats["fast_out_share"] = (fast / tot).reindex(feats.index)
    feats["median_lag_days"] = m[has_in].groupby("node").lag.median()
    # исходящие раньше любых входящих → источник денег вне выборки
    feats["out_before_any_in_kzt"] = m[~has_in].groupby("node").out_amt.sum()

    # синхронность: максимум разных плательщиков в один день
    sync = inc.groupby(["node", "day"]).src.nunique()
    feats["max_payers_same_day"] = sync.groupby(level=0).max()
    feats["sync_days"] = (sync >= 3).groupby(level=0).sum()

    feats = feats.reset_index().rename(columns={"index": "gid"})
    df = df.merge(feats, on="gid", how="left")
    df["days_left_after_last_in"] = last_day - df.last_in_day
    for c in ["fast_out_share", "max_payers_same_day", "sync_days", "out_before_any_in_kzt", "active_in_days"]:
        df[c] = df[c].fillna(0)
    return df


def anomalies(tx: pd.DataFrame, df: pd.DataFrame, th) -> pd.DataFrame:
    inc = tx.rename(columns={"dst": "node"})
    band = inc[(inc.sum_kzt >= th.structuring_lo) & (inc.sum_kzt < th.structuring_hi)]
    n_band = band.groupby("node").size()
    n_all = inc.groupby("node").size()
    df["near_threshold_tx"] = df.gid.map(n_band).fillna(0).astype(int)
    share = (df.near_threshold_tx / df.gid.map(n_all).fillna(0).replace(0, np.nan)).fillna(0)
    df["near_threshold_share"] = share
    df["flag_structuring"] = (df.near_threshold_tx >= 4) & (share >= th.structuring_min_share)

    rep = inc.assign(r=(inc.sum_kzt / 100).round()).groupby(["node", "r"]).size()
    df["max_repeat_amount"] = df.gid.map(rep.groupby(level=0).max()).fillna(0).astype(int)

    # робастный z-score внутри своего колена: «ведёт себя не как соседи по уровню»
    cols = {"log_in": np.log1p(df.in_kzt), "log_out": np.log1p(df.out_kzt),
            "in_deg": df.in_deg.astype(float), "out_deg": df.out_deg.astype(float)}
    zmax = pd.Series(0.0, index=df.index)
    zarg = pd.Series("", index=df.index)
    for name, x in cols.items():
        z = pd.Series(0.0, index=df.index)
        for d, idx in df.groupby("depth").groups.items():
            xs = x.loc[idx]
            med = xs.median()
            mad = (xs - med).abs().median() * 1.4826
            scale = mad if mad > 0 else (xs.std() or 1.0)
            z.loc[idx] = (xs - med) / scale
        better = z.abs() > zmax
        zmax = zmax.where(~better, z.abs())
        zarg = zarg.where(~better, name)
    df["anomaly_z"] = zmax.round(2)
    df["anomaly_feature"] = zarg
    df["flag_anomaly"] = df.anomaly_z >= th.anomaly_z
    return df


def fmt_kzt(x: float) -> str:
    x = float(x or 0)
    if x >= 1e6:
        return f"{x / 1e6:.1f} млн"
    if x >= 1e3:
        return f"{x / 1e3:.0f} тыс."
    return f"{x:.0f}"
