"""Запись выгрузок строго по схеме ТЗ + механическая проверка схемы + данные для экрана просмотра."""
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .config import ROLES

NODE_COLS = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]
CLUSTER_COLS = ["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"]
TOP_COLS = ["rank", "gid", "role", "priority_score", "why"]

EXTRA_NODE_COLS = [
    "is_seed", "depth", "component", "in_deg", "out_deg", "in_kzt", "out_kzt", "in_tx", "out_tx",
    "pass_through", "fan_ratio", "seed_reach", "merge_gain", "n_agg_payers", "n_seed_payers", "seed_traced_in_kzt",
    "pagerank", "hub", "authority", "betweenness", "truncated_by_depth", "p_continue", "terminal_kind",
    "fast_out_share", "median_lag_days", "max_payers_same_day", "near_threshold_tx", "flag_structuring",
    "n_cycles", "anomaly_z", "anomaly_feature", "flag_anomaly", "coord_criteria",
    "prio_role", "prio_traced", "prio_merge", "prio_reach", "prio_pr", "prio_btw", "prio_flags",
]


def write_all(out: Path, df, clusters, top, extras: dict):
    out.mkdir(parents=True, exist_ok=True)
    nr = df[NODE_COLS + EXTRA_NODE_COLS].copy()
    nr["gid"] = nr.gid.astype("int64")
    nr["cluster_id"] = nr.cluster_id.astype(int)
    nr.to_csv(out / "nodes_roles.csv", index=False)
    clusters.to_csv(out / "clusters.csv", index=False)
    top.to_csv(out / "top_nodes.csv", index=False)
    for name, t in extras.items():
        t.to_csv(out / f"{name}.csv", index=False)


def validate(out: Path, n_nodes_expected: int, all_gids: set):
    """То же, что жюри проверит руками. Падает громко, если что-то не так."""
    nr = pd.read_csv(out / "nodes_roles.csv")
    cl = pd.read_csv(out / "clusters.csv")
    tp = pd.read_csv(out / "top_nodes.csv")
    errs = []
    if len(nr) != n_nodes_expected:
        errs.append(f"nodes_roles: {len(nr)} строк вместо {n_nodes_expected}")
    if set(nr.gid) != all_gids:
        errs.append("nodes_roles: набор gid не совпадает с nodes.parquet")
    if nr.gid.duplicated().any():
        errs.append("nodes_roles: дубли gid")
    for c in NODE_COLS:
        if c not in nr or nr[c].isna().any():
            errs.append(f"nodes_roles: пустые значения в {c}")
    if not nr.role.isin(ROLES).all():
        errs.append(f"nodes_roles: роли вне словаря {set(nr.role) - set(ROLES)}")
    if (nr.evidence.astype(str).str.len() == 0).any() or (nr.evidence.astype(str).str.len() > 200).any():
        errs.append("nodes_roles: evidence пустой или > 200 символов")
    if not nr.evidence.astype(str).str.contains(r"\d").all():
        errs.append("nodes_roles: evidence без чисел")
    for c in ["role_score", "priority_score"]:
        if not nr[c].between(0, 1).all():
            errs.append(f"nodes_roles: {c} вне [0,1]")
    if list(cl.columns[:6]) != CLUSTER_COLS or cl[CLUSTER_COLS].isna().any().any():
        errs.append("clusters: схема/пустые значения")
    if not set(nr.cluster_id) <= set(cl.cluster_id):
        errs.append("clusters: есть cluster_id без строки в clusters.csv")
    if len(tp) < 20 or list(tp.columns[:5]) != TOP_COLS:
        errs.append("top_nodes: < 20 строк или неверная схема")
    if not tp.priority_score.is_monotonic_decreasing:
        errs.append("top_nodes: не отсортирован по priority_score")
    if errs:
        raise AssertionError("ПРОВЕРКА СХЕМЫ НЕ ПРОЙДЕНА:\n  " + "\n  ".join(errs))
    print(f"  схема выгрузок: OK (nodes_roles={len(nr)}, clusters={len(cl)}, top_nodes={len(tp)})")


# ---------------------------------------------------------------- данные для экрана просмотра

def _fr_layout(H, iterations=80, seed=42):
    """Векторизованный Fruchterman–Reingold на numpy (в networkx для >500 узлов он в 10–20 раз медленнее)."""
    nodes = list(H.nodes)
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    A = np.zeros((n, n), dtype=np.float32)
    for u, v in H.edges:
        A[idx[u], idx[v]] = A[idx[v], idx[u]] = 1.0
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 2)).astype(np.float32)
    k = 1.0 / np.sqrt(n)
    t = 0.1
    dt = t / (iterations + 1)
    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((delta ** 2).sum(-1))
        np.clip(dist, 0.01, None, out=dist)
        force = (k * k / dist ** 2 - A * dist / k)
        disp = (delta * force[:, :, None]).sum(1)
        length = np.sqrt((disp ** 2).sum(1))
        length = np.where(length < 0.01, 0.1, length)
        pos += disp * (t / length)[:, None]
        t -= dt
    pos -= pos.mean(0)
    pos /= (np.abs(pos).max() or 1)
    return {v: (float(pos[i, 0]), float(pos[i, 1])) for v, i in idx.items()}


def _layout(G, df):
    """Двухуровневая раскладка (быстро и читаемо): кластеры раскладываются как «мета-граф»
    (вес связи = число рёбер между кластерами), узлы — внутри своего кластера."""
    cl = dict(zip(df.gid, df.cluster_id))
    M = nx.Graph()
    M.add_nodes_from(set(cl.values()))
    for u, v in G.edges:
        a, b = cl[u], cl[v]
        if a != b:
            M.add_edge(a, b)
    sizes = df.cluster_id.value_counts().to_dict()
    centers = _fr_layout(M.subgraph([c for c in M if c != 0]), iterations=150) if len(M) > 2 else {}
    if centers:   # разреженные выбросы не должны «сжимать» остальную сеть в точку
        arr = np.array(list(centers.values()))
        q = np.quantile(np.abs(arr), 0.9) or 1.0
        centers = {c: (float(np.clip(x / q, -1.3, 1.3)), float(np.clip(y / q, -1.3, 1.3))) for c, (x, y) in centers.items()}
    span = 110 * np.sqrt(len(df))
    pos = {}
    for c, members in df.groupby("cluster_id").gid:
        members = list(members)
        if c == 0 or c not in centers:          # вне сети — строка внизу
            continue
        cx, cy = centers[c]
        r = 45 * np.sqrt(sizes[c])
        H = G.subgraph(members).to_undirected()
        p = _fr_layout(H, iterations=60) if len(members) > 1 else {members[0]: (0.0, 0.0)}
        for n, (x, y) in p.items():
            pos[n] = (cx * span / 2 + x * r, cy * span / 2 + y * r)
    xs = [x for x, _ in pos.values()] or [0]
    ys = [y for _, y in pos.values()] or [0]
    base = max(ys) + 120
    for i, n in enumerate(df.loc[~df.gid.isin(pos.keys()), "gid"]):
        pos[n] = (min(xs) + 40 * (i % 40), base + 40 * (i // 40))
    return pos


def viewer_payload(G, df, clusters, top, reports: dict, tx=None):
    pos = _layout(G, df)
    keep = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence", "is_seed", "depth",
            "in_deg", "out_deg", "in_kzt", "out_kzt", "pass_through", "seed_reach", "merge_gain",
            "n_seed_payers", "seed_traced_in_kzt", "pagerank", "betweenness", "truncated_by_depth",
            "p_continue", "fast_out_share", "median_lag_days", "max_payers_same_day", "near_threshold_tx",
            "n_cycles", "anomaly_z", "flag_anomaly", "flag_structuring", "component", "n_agg_payers",
            "coord_criteria", "terminal_kind", "out_before_any_in_kzt",
            "prio_role", "prio_traced", "prio_merge", "prio_reach", "prio_pr", "prio_btw", "prio_flags"]
    nd = df[keep].copy()
    nd["x"] = nd.gid.map(lambda g: round(pos[g][0], 1))
    nd["y"] = nd.gid.map(lambda g: round(pos[g][1], 1))
    nd = nd.replace({np.nan: None})
    # gid — 18-значные числа > 2^53: в JavaScript они теряют точность, поэтому в JSON это строки
    nd["gid"] = nd.gid.astype(str)
    edges = [[str(u), str(v), round(d["sum_kzt"], 0), int(d["n_tx"])] for u, v, d in G.edges(data=True)]
    top = top.assign(gid=top.gid.astype(str))
    for key in ("gaps", "routes"):
        for row in reports.get(key, []):
            for c in ("gid", "a", "b", "c"):
                if c in row and row[c] is not None:
                    row[c] = str(row[c])
    return {
        "nodes": nd.to_dict(orient="records"),
        "edges": edges,
        "clusters": clusters.replace({np.nan: None}).to_dict(orient="records"),
        "top": top[["rank", "gid", "role", "priority_score", "why"]].to_dict(orient="records"),
        "tx": [] if tx is None else [[str(a), str(b), int(d), round(float(s), 0)] for a, b, d, s in
                                     zip(tx.src, tx.dst, (tx.date - tx.date.min()).dt.days, tx.sum_kzt)],
        "period_start": None if tx is None else str(tx.date.min().date()),
        **reports,
    }


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    raise TypeError(type(o))


def write_viewer(out: Path, payload: dict, root: Path):
    js = json.dumps(payload, ensure_ascii=False, default=_default)
    (out / "graph.json").write_text(js, encoding="utf-8")
    tpl = (root / "gm" / "viewer_template.html").read_text(encoding="utf-8")
    lib = (root / "vendor" / "cytoscape.min.js").read_text(encoding="utf-8")
    html = tpl.replace("/*__CYTOSCAPE__*/", lib).replace("/*__DATA__*/null", js.replace("</", "<\\/"))
    (out / "viewer.html").write_text(html, encoding="utf-8")
