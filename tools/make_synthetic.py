#!/usr/bin/env python3
"""
Синтетические данные в схеме кейса «Граф денег» — ТОЛЬКО для отладки пайплайна
до выдачи настоящего датасета. На реальных данных этот скрипт не нужен.

Как устроено:
  1. строится «мир» (~20 000 клиентов) с заложенными ролями:
     seed-курьеры → консолидаторы → транзит → координатор → распределители → дропы,
     плюс «бытовые» переводы между случайными людьми;
  2. генерируются транзакции за июль 2026, всё < 5 000 KZT отбрасывается;
  3. от 81 seed делается обход ТОЛЬКО по исходящим на 4 колена — ровно так,
     как собран настоящий граф (поэтому артефакт обрыва на 4-м колене воспроизводится);
  4. пишутся edges / nodes / transactions (+ ground_truth.csv с заложенными ролями
     и истинным наличием исходящих у узлов 4-го колена — для самопроверки).

Запуск:  python tools/make_synthetic.py --out data_synth
"""
import argparse
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import sys

# Windows-консоль (cp1251/cp866) не умеет печатать «≈», «→» и т.п. — принудительно UTF-8
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

RNG = np.random.default_rng(7)
START = pd.Timestamp("2026-07-01")


def day(lo=0, hi=30):
    return START + pd.Timedelta(days=int(RNG.integers(lo, hi + 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_synth")
    ap.add_argument("--csv", action="store_true", help="писать CSV вместо parquet")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    N = 12000
    ids = RNG.choice(np.arange(10_000_000, 99_999_999), size=N, replace=False)
    ptr = iter(range(N))
    take = lambda k=1: [int(ids[next(ptr)]) for _ in range(k)]

    tx = []            # (src, dst, date, sum)
    truth = {}         # gid -> planted role

    def send(s, d, amount, when):
        tx.append((s, d, when, round(float(amount), 2)))

    seeds = take(81)
    for s in seeds:
        truth[s] = "seed"
    rest_pool_start = None

    cells = []
    seed_iter = iter(seeds)
    # 8 ячеек по 8 seed, оставшиеся 17 seed — «одиночки»
    for c in range(8):
        cell_seeds = [next(seed_iter) for _ in range(8)]
        cons = take(2)
        trans = take(int(RNG.integers(0, 2)))   # у половины ячеек транзитного звена нет
        coord = take(1)[0]
        dists = take(int(RNG.integers(1, 3)))
        for x in cons: truth[x] = "consolidator"
        for x in trans: truth[x] = "transit"
        truth[coord] = "coordinator"
        for x in dists: truth[x] = "distributor"
        cells.append(dict(seeds=cell_seeds, cons=cons, trans=trans, coord=coord, dists=dists))
    lone_seeds = list(seed_iter)

    public = [int(x) for x in ids[next(ptr):]]   # «обычные люди»
    pub = np.array(public)
    # латентная «активность»: активные чаще получают И чаще отправляют —
    # именно эту связь потом использует модель обрыва 4-го колена
    act = RNG.lognormal(0, 1.0, len(pub))
    p_pick = act / act.sum()
    pick = lambda k=None: RNG.choice(pub, size=k, p=p_pick) if k else int(RNG.choice(pub, p=p_pick))

    # --- преступная схема
    for ci, cell in enumerate(cells):
        cons_in = {x: 0.0 for x in cell["cons"]}
        for s in cell["seeds"]:
            targets = cell["cons"] if RNG.random() < 0.5 else [cell["cons"][int(RNG.integers(0, 2))]]
            for c in targets:
                for _ in range(int(RNG.integers(2, 6))):
                    amt = RNG.uniform(40_000, 350_000)
                    send(s, c, amt, day(0, 25))
                    cons_in[c] += amt
        # структурирование: один консолидатор ячейки 0 получает дроблёные суммы чуть выше порога
        if ci == 0:
            for s in cell["seeds"]:
                for _ in range(6):
                    amt = RNG.uniform(5_000, 9_900)
                    send(s, cell["cons"][0], amt, day(3, 20))
                    cons_in[cell["cons"][0]] += amt
        # консолидаторы → транзит (или сразу координатору) через 0–2 дня
        for c, total in cons_in.items():
            nxt = (cell["trans"] + [cell["coord"]])[0]
            for part in np.array_split(np.arange(4), 4):
                send(c, nxt, total * RNG.uniform(0.18, 0.23), day(5, 28))
        # транзит → транзит → координатор, пропуск 0.9–1.05
        chain = cell["trans"] + [cell["coord"]]
        for a_, b_ in zip(chain[:-1], chain[1:]):
            send(a_, b_, sum(cons_in.values()) * RNG.uniform(0.85, 0.98) * 0.5, day(20, 29))
            send(a_, b_, sum(cons_in.values()) * RNG.uniform(0.85, 0.98) * 0.4, day(22, 30))
        # координатор → распределители (60%), часть оседает
        for d in cell["dists"]:
            send(cell["coord"], d, sum(cons_in.values()) * 0.6 / len(cell["dists"]), day(24, 30))
        # распределители → 50–110 дропов
        for d in cell["dists"]:
            k = int(RNG.integers(50, 111))
            for r in RNG.choice(pub, size=k, replace=False):  # дропы — случайные люди
                send(d, int(r), RNG.uniform(6_000, 60_000), day(25, 30))
    # связь между ячейками: координаторы 0,1,2 отдают «наверх» одному узлу
    boss = take(1)[0]
    truth[boss] = "coordinator"
    for cell in cells[:3]:
        send(cell["coord"], boss, RNG.uniform(3e6, 6e6), day(26, 30))
    # возвратный поток: босс возвращает часть одному из seed ячейки 0
    send(boss, cells[0]["seeds"][0], 800_000, day(29, 30))

    # --- бытовые переводы
    for s in seeds:
        for _ in range(int(RNG.integers(0, 9))):
            send(s, pick(), RNG.lognormal(9.5, 1.0), day())
    # 25 seed «молчат» (переводы < 5000, отсекутся порогом)
    for s in lone_seeds[:10]:
        for _ in range(3):
            send(s, int(RNG.choice(pub)), RNG.uniform(500, 4_900), day())
    # у обычных людей вероятность иметь исходящие растёт с активностью
    p_out = 1 / (1 + np.exp(-(np.log(act) * 1.6 - 0.2)))
    has_out = RNG.random(len(pub)) < p_out
    for p, h, a_ in zip(pub, has_out, act):
        if h:
            for _ in range(int(RNG.integers(1, 3 + min(int(a_), 3)))):
                send(int(p), pick(), RNG.lognormal(9.8, 1.1), day())

    T = pd.DataFrame(tx, columns=["src", "dst", "date", "sum_kzt"])
    T = T[T.src != T.dst]
    T = T[T.sum_kzt >= 5_000].reset_index(drop=True)       # порог выгрузки

    # --- обход от seed только по исходящим, 4 колена
    adj = T.groupby("src").dst.apply(lambda s: sorted(set(s))).to_dict()
    depth = {s: 0 for s in seeds}
    q = deque(seeds)
    while q:
        u = q.popleft()
        if depth[u] >= 4:
            continue
        for v in adj.get(u, []):
            if v not in depth:
                depth[v] = depth[u] + 1
                q.append(v)
    keep = T[T.src.map(lambda x: depth.get(x, 99) <= 3) & T.dst.isin(depth.keys())].copy()
    nodes = pd.DataFrame({"gid": list(depth.keys()), "depth": list(depth.values())})
    nodes["is_seed"] = nodes.gid.isin(seeds)
    edges = keep.groupby(["src", "dst"]).agg(sum_kzt=("sum_kzt", "sum"), n_tx=("sum_kzt", "size")).reset_index()
    edges["depth"] = edges.src.map(depth) + 1
    keep = keep.sort_values("date").reset_index(drop=True)

    gt = nodes[["gid", "depth"]].copy()
    gt["planted_role"] = gt.gid.map(truth).fillna("public")
    gt["true_has_out"] = gt.gid.isin(set(T.src))   # включая переводы за пределами обхода

    for name, df in [("edges", edges), ("nodes", nodes), ("transactions", keep)]:
        if a.csv:
            df.to_csv(out / f"{name}.csv", index=False)
        else:
            df.to_parquet(out / f"{name}.parquet", index=False)
    gt.to_csv(out / "ground_truth.csv", index=False)
    print(f"nodes={len(nodes)} edges={len(edges)} tx={len(keep)} "
          f"by depth={nodes.depth.value_counts().sort_index().to_dict()} "
          f"turnover={edges.sum_kzt.sum():,.0f}")


if __name__ == "__main__":
    main()
