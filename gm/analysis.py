"""Дополнительные анализы: циклы, повторяющиеся маршруты, устойчивость сети, белые пятна."""
import networkx as nx
import numpy as np
import pandas as pd

from .features import fmt_kzt


# ---------------------------------------------------------------- циклы (возвратные потоки)

def cycles(G, df, length_bound=6, limit=5000):
    rows = []
    for cyc in nx.simple_cycles(G, length_bound=length_bound):
        pairs = list(zip(cyc, cyc[1:] + cyc[:1]))
        flow = min(G[u][v]["sum_kzt"] for u, v in pairs)
        rows.append({"cycle": "→".join(map(str, cyc + [cyc[0]])), "length": len(cyc),
                     "bottleneck_kzt": round(flow, 2), "nodes": cyc})
        if len(rows) >= limit:
            break
    cyc_df = pd.DataFrame(rows, columns=["cycle", "length", "bottleneck_kzt", "nodes"])
    cnt = {}
    for ns in cyc_df.nodes:
        for n in ns:
            cnt[n] = cnt.get(n, 0) + 1
    df["n_cycles"] = df.gid.map(cnt).fillna(0).astype(int)
    return df, cyc_df.drop(columns="nodes").sort_values("bottleneck_kzt", ascending=False)


# ---------------------------------------------------------------- повторяющиеся маршруты A→B→C

def routes(G, tx, top=100):
    """Устойчивые двухзвенные маршруты: через B прошло ≥2 перевода и деньги ушли после поступления."""
    t = tx.copy()
    rows = []
    for b in G.nodes:
        ins, outs = list(G.in_edges(b, data=True)), list(G.out_edges(b, data=True))
        if not ins or not outs:
            continue
        for a, _, di in ins:
            for _, c, do in outs:
                if a == c:
                    continue
                flow = min(di["sum_kzt"], do["sum_kzt"])
                rows.append((a, b, c, flow, di["n_tx"] + do["n_tx"]))
    r = pd.DataFrame(rows, columns=["a", "b", "c", "bottleneck_kzt", "n_tx"])
    if r.empty:
        return r
    r = r.sort_values("bottleneck_kzt", ascending=False).head(top * 3)
    first_in = t.groupby(["src", "dst"]).date.min()
    last_out = t.groupby(["src", "dst"]).date.max()
    r["lag_days"] = [(last_out.get((b, c)) - first_in.get((a, b))).days for a, b, c in zip(r.a, r.b, r.c)]
    r = r[r.lag_days >= 0].head(top)
    r["route"] = r.a.astype(str) + "→" + r.b.astype(str) + "→" + r.c.astype(str)
    return r[["route", "a", "b", "c", "bottleneck_kzt", "n_tx", "lag_days"]]


# ---------------------------------------------------------------- устойчивость

def _reach_pairs(G, seeds):
    return sum(len(nx.descendants(G, s)) for s in seeds if s in G)


def resilience(G, df, ns=(0, 5, 10, 20, 50), random_runs=10, rng_seed=0):
    """Что будет с сетью, если изъять топ-N узлов. Метрика — доля пар (seed → узел), между
    которыми остаётся денежный путь: насколько «перерезаны» каналы движения денег."""
    seeds = list(df.loc[df.is_seed, "gid"])
    base = _reach_pairs(G, seeds) or 1
    order = {
        "priority": list(df.sort_values("priority_score", ascending=False).gid),
        "pagerank": list(df.sort_values("pagerank", ascending=False).gid),
        "in_kzt": list(df.sort_values("in_kzt", ascending=False).gid),
    }
    non_seed = list(df.loc[~df.is_seed, "gid"])
    rng = np.random.default_rng(rng_seed)
    rows = []
    for n in ns:
        for name, lst in order.items():
            rm = [x for x in lst if x not in set(seeds)][:n]      # seed не изымаем — они уже известны
            H = G.copy()
            H.remove_nodes_from(rm)
            rows.append(_res_row(H, seeds, base, name, n))
        vals = []
        for _ in range(random_runs):
            rm = list(rng.choice(non_seed, size=min(n, len(non_seed)), replace=False)) if n else []
            H = G.copy()
            H.remove_nodes_from(rm)
            vals.append(_res_row(H, seeds, base, "random", n))
        avg = {k: (np.mean([v[k] for v in vals]) if isinstance(vals[0][k], (int, float)) else vals[0][k])
               for k in vals[0]}
        rows.append(avg)
    return pd.DataFrame(rows)


def _res_row(H, seeds, base, strategy, n):
    comps = [c for c in nx.weakly_connected_components(H) if len(c) > 1]
    largest = max((len(c) for c in comps), default=0)
    return {"strategy": strategy, "removed": n,
            "seed_paths_left_share": round(_reach_pairs(H, seeds) / base, 4),
            "components_2plus": len(comps), "largest_component": largest}


# ---------------------------------------------------------------- белые пятна / следующий запрос

def gaps(df, top_k=15):
    rows = []
    tr = df[df.truncated_by_depth & (df.p_continue >= 0.5)].sort_values("in_kzt", ascending=False).head(top_k)
    for r in tr.itertuples(index=False):
        rows.append((r.gid, "обрыв 4-го колена",
                     f"получил {fmt_kzt(r.in_kzt)}, P(ушли дальше)={r.p_continue:.2f}",
                     "Выгрузить исходящие переводы (5-е колено)"))
    top = df.sort_values("priority_score", ascending=False).head(top_k)
    for r in top.itertuples(index=False):
        if r.out_before_any_in_kzt > 0 or (not pd.isna(r.pass_through) and r.pass_through > 1.2):
            rows.append((r.gid, "источник вне выборки",
                         f"отдал {fmt_kzt(r.out_kzt)} при входе {fmt_kzt(r.in_kzt)}",
                         "Выгрузить ВСЕ входящие переводы клиента за период (не только от графа)"))
    st = df[df.flag_structuring]
    for r in st.itertuples(index=False):
        rows.append((r.gid, "возможное дробление",
                     f"{r.near_threshold_tx} входящих в диапазоне 5–10 тыс.",
                     "Запросить транзакции < 5 000 KZT (ниже порога выгрузки)"))
    term = df[(df.role == "terminal") & (df.terminal_kind == "observed")].sort_values("in_kzt", ascending=False).head(top_k)
    for r in term.itertuples(index=False):
        rows.append((r.gid, "деньги «осели»",
                     f"получил {fmt_kzt(r.in_kzt)}, внутрибанковских исходящих нет",
                     "Проверить межбанковские переводы, снятие наличных, карточные операции"))
    iso = df[(df.in_deg == 0) & (df.out_deg == 0)]
    if len(iso):
        rows.append(("—", "seed вне сети", f"{len(iso)} клиентов без переводов ≥5 000 внутри банка",
                     "Запросить межбанк/наличные и операции < 5 000 KZT по этим seed"))
    return pd.DataFrame(rows, columns=["gid", "gap", "observation", "next_request"])
