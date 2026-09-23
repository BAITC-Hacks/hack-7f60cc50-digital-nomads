#!/usr/bin/env python3
"""
«Граф денег» — полный пересчёт от сырых parquet до выгрузок одной командой.

    python run.py --data data --out out

На выходе (в --out):
  nodes_roles.csv, clusters.csv, top_nodes.csv  — обязательные выгрузки по схеме ТЗ
  cycles.csv, routes.csv, resilience.csv, gaps.csv — дополнительные анализы
  run_report.json, report.md                      — качество данных, пороги, модель обрыва, тайминги
  viewer.html                                     — экран просмотра (открывается двойным кликом, офлайн)
  cases/case_<gid>.md                             — досье на топ-узлы: факты, пути денег с датами, что запросить
  graph.json                                      — данные для AI-ассистента (serve.py)
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Windows-консоль (cp1251/cp866) не умеет печатать «≈», «→» и т.п. — принудительно UTF-8
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from gm import analysis, boundary, clusters, export, features, investigate, io, priority, roles  # noqa: E402
from gm.config import ROLE_RU, Thresholds  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data", help="папка с edges/nodes/transactions.parquet")
    ap.add_argument("--out", default="out", help="куда писать выгрузки")
    ap.add_argument("--config", default=None, help="JSON с переопределением порогов (gm/config.py)")
    ap.add_argument("--no-viewer", action="store_true", help="не строить viewer.html и досье")
    ap.add_argument("--cases", type=int, default=5, help="сколько досье на топ-узлы сформировать в out/cases/")
    a = ap.parse_args()

    th = Thresholds.load(a.config)
    out = Path(a.out)
    T, t0 = {}, time.perf_counter()

    def tick(name):
        nonlocal t0
        T[name] = round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        print(f"  [{T[name]:>5.2f}s] {name}")

    edges, nodes, tx = io.load(a.data)
    dq = io.sanity_check(edges, nodes, tx)
    tick("загрузка и проверка данных")

    G = features.build_graph(edges, nodes)
    df = features.structural(G, nodes)
    tick("граф и структурные метрики")
    df = features.provenance(G, df, tx)
    tick("происхождение денег (seed_reach, merge_gain, трассировка)")
    df = features.temporal(tx, df, th.fast_days)
    df = features.anomalies(tx, df, th)
    tick("временные паттерны и аномалии")
    df, bmodel = boundary.fit_boundary_model(df, tx)
    bmodel["terminal_threshold"] = boundary.threshold_summary(bmodel, th.terminal_max_p_continue)
    tick("модель обрыва 4-го колена")
    df, cyc = analysis.cycles(G, df)
    rts = analysis.routes(G, tx)
    tick("циклы и маршруты")

    df = roles.assign_roles(G, df, th)
    tick("роли, role_score, evidence")
    df, stab = clusters.cluster(G, df, th)
    tick("кластеризация Louvain + устойчивость")
    df = priority.prioritize(df, th)
    top = priority.top_table(df, max(20, th.top_n))
    ctab = clusters.cluster_table(G, df, stab)
    tick("приоритеты, топ-лист, таблица кластеров")
    res = analysis.resilience(G, df)
    gp = analysis.gaps(df, th)
    tick("устойчивость сети и белые пятна")

    export.write_all(out, df, ctab, top, {"cycles": cyc, "routes": rts, "resilience": res, "gaps": gp})
    export.validate(out, len(nodes), set(nodes.gid))
    tick("запись и проверка выгрузок")

    role_counts = df.role.value_counts().to_dict()
    report = {
        "data_quality": dq,
        "thresholds": th.as_dict(),
        "boundary_model": bmodel,
        "role_counts": role_counts,
        "n_clusters": int(ctab.cluster_id.nunique()),
        "clusters_multi_seed": int((ctab.n_seed > 1).sum()),
        "n_cycles": int(len(cyc)),
        "timings_sec": T,
    }
    if not a.no_viewer:
        payload = export.viewer_payload(G, df, ctab, top, {
            "report": report, "resilience": res.to_dict(orient="records"),
            "gaps": gp.to_dict(orient="records"), "routes": rts.head(50).to_dict(orient="records"),
            "cycles": cyc.head(50).to_dict(orient="records"),
        }, tx=tx)
        export.write_viewer(out, payload, ROOT)
        tick("экран просмотра viewer.html")
        # досье на топ-N без LLM: жюри читает их сразу, без сервера (python investigate.py — любой gid)
        cases = investigate.write_cases(out, json.loads((out / "graph.json").read_text(encoding="utf-8")), a.cases)
        tick(f"досье на топ-{len(cases)} (out/cases/)")
    report["timings_sec"] = T
    report["total_sec"] = round(sum(T.values()), 2)
    (out / "run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=export._default),
                                         encoding="utf-8")
    _markdown_report(out, report, top, ctab, res)

    print("\nРОЛИ:", ", ".join(f"{ROLE_RU[k]}={v}" for k, v in role_counts.items()))
    if "cv_auc_mean" in bmodel:
        sh = bmodel.get("shift_auc_depth12_to_3")
        print(f"МОДЕЛЬ ОБРЫВА [{bmodel['status']}]: CV AUC={bmodel['cv_auc_mean']:.3f}, "
              f"перенос 1–2→3 колено AUC={sh:.3f}, " if sh is not None else
              f"МОДЕЛЬ ОБРЫВА [{bmodel['status']}]: CV AUC={bmodel['cv_auc_mean']:.3f}, ", end="")
        print(f"ожидаемо настоящих стоков среди {bmodel['n_truncated']} обрезанных ≈ {bmodel['expected_true_terminals']:.0f}")
    print(f"КЛАСТЕРОВ: {report['n_clusters']} (с >1 seed: {report['clusters_multi_seed']}); циклов: {len(cyc)}")
    print(f"ГОТОВО за {report['total_sec']} с → {out.resolve()}")
    if not a.no_viewer:
        print(f"Экран просмотра: {(out / 'viewer.html').resolve()}  (AI-ассистент: python serve.py --out {out})")


def _markdown_report(out, rep, top, ctab, res):
    L = ["# Отчёт прогона", "", "## Качество данных", ""]
    for k, v in rep["data_quality"].items():
        L.append(f"- **{k}**: {v}")
    L += ["", "## Роли", ""] + [f"- {ROLE_RU[k]} (`{k}`): {v}" for k, v in rep["role_counts"].items()]
    b = rep["boundary_model"]
    L += boundary.calibration_markdown(b, rep["thresholds"]["terminal_max_p_continue"])
    L += ["", f"Ожидаемое число настоящих стоков среди {b['n_truncated']} обрезанных: {b['expected_true_terminals']:.0f}."]
    svg = boundary.calibration_svg(b)
    if svg:
        (out / "calibration.svg").write_text(svg, encoding="utf-8")
    L += ["", "## Топ-10", "", "| # | gid | роль | priority | почему |", "|---|---|---|---|---|"]
    for r in top.head(10).itertuples(index=False):
        L.append(f"| {r.rank} | {r.gid} | {r.role} | {r.priority_score:.3f} | {r.why[:160]} |")
    L += ["", "## Кластеры (первые 10)", "", "| id | узлов | seed | оборот внутри | устойчивость | гипотеза |",
          "|---|---|---|---|---|---|"]
    for r in ctab.head(10).itertuples(index=False):
        L.append(f"| {r.cluster_id} | {r.n_nodes} | {r.n_seed} | {r.sum_kzt_internal:,.0f} | {r.stability} | {r.hypothesis} |")
    strat = list(dict.fromkeys(res.strategy))
    L += ["", "## Устойчивость: доля путей seed→узел после изъятия N узлов", "",
          "| N | " + " | ".join(strat) + " |", "|---" * (len(strat) + 1) + "|"]
    for n, g in res.groupby("removed"):
        d = dict(zip(g.strategy, g.seed_paths_left_share))
        L.append(f"| {int(n)} | " + " | ".join(f"{d.get(s, 0):.2f}" for s in strat) + " |")
    L += ["", f"Время полного прогона: {rep['total_sec']} с"]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
