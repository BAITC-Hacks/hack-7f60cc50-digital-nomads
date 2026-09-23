"""
Кластеризация: Louvain на НЕориентированной проекции (оговорено явно — метод не умеет направление).
Вес ребра = log1p(сумма u→v + v→u): одна гигантская сумма не должна «перетягивать» сообщество,
важна плотность денежных связей. Направление потоков сохраняется в метриках и гипотезах кластера.

Устойчивость: Louvain перезапускается с 5 разными random seed; для каждого кластера считаем
средний лучший Жаккар с кластерами других прогонов (1.0 = кластер воспроизводится всегда).
Узлы без рёбер (seed без переводов ≥5 000) → cluster_id = 0 «вне сети».
"""
import networkx as nx
import numpy as np
import pandas as pd

from .config import ROLE_RU
from .features import fmt_kzt


def _undirected(G):
    UG = nx.Graph()
    UG.add_nodes_from(G.nodes)
    for u, v, d in G.edges(data=True):
        w = d["sum_kzt"]
        if UG.has_edge(u, v):
            UG[u][v]["raw"] += w
        else:
            UG.add_edge(u, v, raw=w)
    for u, v, d in UG.edges(data=True):
        d["weight"] = float(np.log1p(d["raw"]))
    return UG


def _louvain(UG, seed, res):
    sub = UG.subgraph([n for n in UG.nodes if UG.degree(n) > 0])
    return nx.community.louvain_communities(sub, weight="weight", resolution=res, seed=seed)


def cluster(G, df, th):
    UG = _undirected(G)
    comms = sorted(_louvain(UG, th.louvain_seed, th.louvain_resolution), key=len, reverse=True)
    cid = {}
    for i, c in enumerate(comms, start=1):
        for n in c:
            cid[n] = i
    df["cluster_id"] = df.gid.map(cid).fillna(0).astype(int)

    # устойчивость
    runs = [_louvain(UG, s, th.louvain_resolution) for s in (1, 2, 3, 4, 5)]
    stab = {0: 1.0}
    for i, c in enumerate(comms, start=1):
        js = []
        for run in runs:
            js.append(max(len(c & o) / len(c | o) for o in run))
        stab[i] = round(float(np.mean(js)), 2)
    return df, stab


def cluster_table(G, df, stab):
    rows = []
    by = df.groupby("cluster_id")
    for c, g in by:
        members = set(g.gid)
        internal = ext_in = ext_out = 0.0
        for u in members:
            for _, v, d in G.out_edges(u, data=True):
                if v in members:
                    internal += d["sum_kzt"]
                else:
                    ext_out += d["sum_kzt"]
            for w, _, d in G.in_edges(u, data=True):
                if w not in members:
                    ext_in += d["sum_kzt"]
        roles = g.role.value_counts().to_dict()
        top = g.sort_values("priority_score", ascending=False).head(5)
        rows.append({
            "cluster_id": int(c),
            "n_nodes": int(len(g)),
            "n_seed": int(g.is_seed.sum()),
            "sum_kzt_internal": round(internal, 2),
            "top_gids": ";".join(str(x) for x in top.gid),
            "hypothesis": _hypothesis(c, g, roles, internal, top, stab.get(int(c), 1.0)),
            "sum_kzt_in_external": round(ext_in, 2),
            "sum_kzt_out_external": round(ext_out, 2),
            "stability": stab.get(int(c), np.nan),
            "roles": ", ".join(f"{ROLE_RU[k]}:{v}" for k, v in sorted(roles.items(), key=lambda x: -x[1])),
            "max_priority": round(float(g.priority_score.max()), 4),
            "components": ",".join(str(x) for x in sorted(g.component.unique())),
        })
    t = pd.DataFrame(rows).sort_values(["max_priority", "n_nodes"], ascending=[False, False])
    return t


def _hypothesis(c, g, roles, internal, top, stability=1.0):
    h = _hypothesis_core(c, g, roles, internal, top)
    if c != 0 and stability < 0.5:
        h += f". Кластер неустойчив (воспроизводимость {stability:.2f}) — границы условны"
    return h


def _hypothesis_core(c, g, roles, internal, top):
    n, ns = len(g), int(g.is_seed.sum())
    if c == 0:
        return (f"Вне сети: {n} клиент(ов) без переводов ≥5 000 KZT внутри банка. "
                f"Гипотеза: наличные, межбанк или дробление ниже порога — нужен отдельный запрос")
    lead = top.iloc[0]
    lead_s = f"{lead.gid} ({ROLE_RU[lead.role]}, №1 по приоритету в кластере)"
    trunc = g.truncated_by_depth.mean()
    tail = f"; {trunc * 100:.0f}% узлов на границе выгрузки" if trunc >= 0.4 else ""
    if roles.get("coordinator") and ns >= 1:
        return (f"Признаки управляющего звена: {ns} seed, ключевой узел {lead_s}; "
                f"внутренний оборот {fmt_kzt(internal)}. Проверить как возможную вершину схемы{tail}")
    if roles.get("coordinator"):
        return (f"Узел концентрации потоков {lead_s} без seed внутри кластера; оборот {fmt_kzt(internal)}. "
                f"Связь с seed только через внешние рёбра — сначала исключить легальный бизнес (KYC){tail}")
    if roles.get("consolidator") and ns >= 2:
        return (f"Ячейка сбора: {ns} seed → {roles['consolidator']} консолидатор(а); {lead_s}; "
                f"оборот {fmt_kzt(internal)}{tail}")
    if roles.get("distributor"):
        return (f"Распределительная ветка: {lead_s}; {roles.get('peripheral', 0)} периферийных "
                f"получателей — возможные дропы/выплаты{tail}")
    if roles.get("transit", 0) >= 2:
        return f"Транзитная цепочка ({roles['transit']} транзитных узлов), ведущий {lead_s}{tail}"
    if g.component.nunique() == 1 and g.component.iloc[0] != 0 and n < 20:
        return f"Изолированный фрагмент ({n} узлов, {ns} seed), вне основной сети{tail}"
    return f"Периферийная ветка без выраженной структуры, ведущий узел {lead_s}{tail}"
