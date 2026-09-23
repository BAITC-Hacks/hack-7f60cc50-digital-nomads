"""
priority_score ∈ [0,1] — «кого смотреть первым». Взвешенная сумма перцентилей, каждая
составляющая объяснима и печатается в top_nodes.why.

  0.25 × вес роли × role_score     — роль как точка вмешательства
  0.20 × pct(seed_traced_in_kzt)   — сколько денег с происхождением от seed проходит через узел
  0.15 × pct(merge_gain)           — узел впервые сводит потоки разных seed
  0.15 × pct(seed_reach)           — от скольких seed к узлу доходят деньги (по хронологии переводов)
  0.10 × pct(pagerank)             — куда стекаются деньги в сети
  0.10 × pct(betweenness)          — посредник на маршрутах
  0.05 × флаги                     — сквозной пропуск, дробление, синхронность, цикл, аномалия
  × seed_priority_factor для seed  — они уже известны; фокус на новых узлах
  × weak_link_factor                — не-seed без прямого плательщика-seed, у которого от seed-цепочки
                                      прослеживается < min_seed_share_top входящих денег: связь слабая,
                                      высок риск ложного срабатывания (например, легальный бизнес)
"""
import numpy as np
import pandas as pd

from .config import ROLE_RU, ROLE_WEIGHT
from .features import fmt_kzt, pct_pos

W = {"role": 0.25, "traced": 0.20, "merge": 0.15, "reach": 0.15, "pr": 0.10, "btw": 0.10, "flags": 0.05}


def prioritize(df: pd.DataFrame, th) -> pd.DataFrame:
    comp = pd.DataFrame(index=df.index)
    comp["role"] = df.role.map(ROLE_WEIGHT) * df.role_score
    comp["traced"] = pct_pos(df.seed_traced_in_kzt)
    comp["merge"] = pct_pos(df.merge_gain)
    comp["reach"] = pct_pos(df.seed_reach)
    comp["pr"] = pct_pos(df.pagerank)
    comp["btw"] = pct_pos(df.betweenness)
    flags = ((df.fast_out_share >= 0.7) & (df.in_deg > 0) & (df.out_deg > 0)).astype(int) \
        + df.flag_structuring.astype(int) + (df.max_payers_same_day >= 3).astype(int) \
        + (df.n_cycles > 0).astype(int) + df.flag_anomaly.astype(int)
    comp["flags"] = (flags / 3).clip(upper=1)
    raw = sum(W[k] * comp[k] for k in W)
    raw = np.where(df.is_seed, raw * th.seed_priority_factor, raw)
    df["seed_share_in"] = (df.seed_traced_in_kzt / df.in_kzt.where(df.in_kzt > 0)).fillna(0).clip(upper=1).round(4)
    weak = (~df.is_seed) & (df.n_seed_payers == 0) & (df.seed_share_in < th.min_seed_share_top)
    df["weak_seed_link"] = weak
    raw = np.where(weak, raw * th.weak_link_factor, raw)
    df["priority_score"] = np.round(raw, 4)
    df["priority_rank"] = df.priority_score.rank(ascending=False, method="first").astype(int)
    for k in W:
        df[f"prio_{k}"] = (W[k] * comp[k]).round(4)
    return df


def top_table(df: pd.DataFrame, n: int) -> pd.DataFrame:
    t = df.sort_values("priority_score", ascending=False).head(n).copy()
    t["rank"] = np.arange(1, len(t) + 1)
    t["why"] = [_why(r) for r in t.itertuples(index=False)]
    cols = ["rank", "gid", "role", "priority_score", "why", "role_score", "cluster_id", "is_seed",
            "seed_reach", "seed_share_in", "pattern", "in_deg", "out_deg", "in_tx", "out_tx", "in_kzt", "out_kzt"]
    return t[cols]


def _why(r) -> str:
    parts = [f"{ROLE_RU[r.role]} (уверенность {r.role_score:.2f}): {r.evidence}"]
    drivers = sorted(
        [("деньги от seed-цепочки " + fmt_kzt(r.seed_traced_in_kzt), r.prio_traced),
         (f"сводит потоки {r.merge_gain} seed", r.prio_merge),
         (f"по хронологии доходят деньги {r.seed_reach} seed", r.prio_reach),
         ("высокий PageRank (сток денег)", r.prio_pr),
         ("посредник на маршрутах (betweenness)", r.prio_btw)],
        key=lambda x: -x[1])
    main = [d for d, v in drivers[:2] if v > 0]
    if main:
        parts.append("Главные факторы приоритета: " + ", ".join(main) + ".")
    if r.weak_seed_link:
        parts.append(f"Связь с seed слабая ({r.seed_share_in * 100:.0f}% входа) — приоритет понижен.")
    if r.pattern != "none":
        parts.append("Типология: " + r.pattern.replace(";", ", ") + ".")
    if r.is_seed:
        parts.append("Уже известный seed-клиент.")
    parts.append(f"Кластер {r.cluster_id}. Гипотеза для проверки, не вывод о виновности.")
    return " ".join(parts)
