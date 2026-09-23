#!/usr/bin/env python3
"""Объяснить роль любого gid через фактические значения и пороги.

Пример:
    python explain.py --gid 100000003115284100 --out out

Команда читает результат пайплайна, ничего не пересчитывает и не использует LLM.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from gm.config import ROLE_RU, Thresholds


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def _n(value):
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f} млн"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f} тыс."
    return f"{value:g}"


def _check(ok, fact, rule):
    return f"{'✓' if ok else '✗'} {fact}; правило: {rule}"


def explain_row(row, all_nodes: pd.DataFrame, th: Thresholds):
    recv = all_nodes.loc[all_nodes.in_kzt > 0, "in_kzt"]
    p75 = float(recv.quantile(th.cons_min_in_pct)) if len(recv) else 0.0
    p90 = float(recv.quantile(th.coord_min_in_pct)) if len(recv) else 0.0
    role = str(row.role)
    pt = float(row.pass_through) if pd.notna(row.pass_through) else 0.0
    lines = [
        f"gid: {row.gid}",
        f"роль: {role} ({ROLE_RU.get(role, role)})",
        f"role_score: {float(row.role_score):.3f}",
        f"priority_score: {float(row.priority_score):.4f} · место №{int(row.priority_rank)}",
        "",
        "Проверка правила:",
    ]

    if role == "coordinator":
        trough = int(row.out_deg) == 1 and th.transit_pass_lo <= pt <= th.transit_pass_hi
        lines += [
            _check(row.seed_reach >= th.coord_min_seed_reach,
                   f"по хронологии достигают {int(row.seed_reach)} seed",
                   f"seed_reach ≥ {th.coord_min_seed_reach}"),
            _check(row.in_kzt >= p90, f"вход {_n(row.in_kzt)}", f"in_kzt ≥ P90 ({_n(p90)})"),
            _check(not trough, f"out_deg={int(row.out_deg)}, пропуск={pt:.2f}", "узел не является жёлобом 1→1"),
            _check(str(row.coord_criteria) not in ("", "none", "nan"),
                   f"структурный критерий: {row.coord_criteria}",
                   "есть хотя бы один критерий кроме одиночного «собирает и рассылает»"),
        ]
    elif role == "consolidator":
        hub = pt > th.cons_hub_pass and row.out_deg >= th.dist_min_receivers
        lines += [
            _check(row.in_deg >= th.cons_min_payers, f"плательщиков {int(row.in_deg)}",
                   f"in_deg ≥ {th.cons_min_payers}"),
            _check(row.in_kzt >= p75, f"вход {_n(row.in_kzt)}", f"in_kzt ≥ P75 ({_n(p75)})"),
            _check(not hub, f"пропуск={pt:.2f}, получателей={int(row.out_deg)}",
                   f"не хаб: не (пропуск > {th.cons_hub_pass} и out_deg ≥ {th.dist_min_receivers})"),
        ]
    elif role == "distributor":
        lines += [
            _check(row.out_deg >= th.dist_min_receivers, f"получателей {int(row.out_deg)}",
                   f"out_deg ≥ {th.dist_min_receivers}"),
            _check(row.fan_ratio >= th.dist_min_fan_ratio, f"веер {float(row.fan_ratio):.2f}:1",
                   f"fan_ratio ≥ {th.dist_min_fan_ratio}"),
        ]
    elif role == "transit":
        lines += [
            _check(not bool(row.is_seed), f"is_seed={bool(row.is_seed)}", "seed не получает роль transit"),
            _check(th.transit_pass_lo <= pt <= th.transit_pass_hi, f"out/in={pt:.2f}",
                   f"{th.transit_pass_lo} ≤ out/in ≤ {th.transit_pass_hi}"),
            _check(row.in_deg <= th.transit_max_degree and row.out_deg <= th.transit_max_degree,
                   f"степени {int(row.in_deg)}→{int(row.out_deg)}",
                   f"in_deg и out_deg ≤ {th.transit_max_degree}"),
            _check(row.in_kzt >= th.transit_min_in_kzt, f"вход {_n(row.in_kzt)}",
                   f"in_kzt ≥ {_n(th.transit_min_in_kzt)}"),
        ]
    elif role == "terminal":
        kind = str(row.terminal_kind)
        big = row.in_kzt >= th.terminal_min_in_kzt or row.in_deg >= 2
        if kind == "observed":
            lines += [
                _check(row.out_deg == 0, f"out_deg={int(row.out_deg)}", "out_deg = 0"),
                _check(row.depth < 4, f"колено {int(row.depth)}", "depth < 4, исходящие наблюдаются"),
                _check(big, f"вход {_n(row.in_kzt)}, плательщиков {int(row.in_deg)}",
                       f"in_kzt ≥ {_n(th.terminal_min_in_kzt)} или in_deg ≥ 2"),
            ]
        elif kind == "partial":
            lines += [
                _check(row.depth < 4, f"колено {int(row.depth)}", "depth < 4"),
                _check(pt <= th.terminal_max_pass, f"дальше ушло {pt * 100:.1f}%",
                       f"out/in ≤ {th.terminal_max_pass}"),
                _check(row.in_kzt >= th.terminal_min_partial_kzt, f"вход {_n(row.in_kzt)}",
                       f"in_kzt ≥ {_n(th.terminal_min_partial_kzt)}"),
            ]
        else:
            lines += [
                _check(bool(row.truncated_by_depth), f"колено {int(row.depth)} обрезано",
                       "truncated_by_depth = true"),
                _check(row.p_continue < th.terminal_max_p_continue,
                       f"P(деньги ушли дальше)={float(row.p_continue):.3f}",
                       f"p_continue < {th.terminal_max_p_continue}"),
                _check(big, f"вход {_n(row.in_kzt)}, плательщиков {int(row.in_deg)}",
                       f"in_kzt ≥ {_n(th.terminal_min_in_kzt)} или in_deg ≥ 2"),
            ]
    else:
        hub = pt > th.cons_hub_pass and row.out_deg >= th.dist_min_receivers
        lines += [
            _check(row.in_deg >= th.cons_min_payers and row.in_kzt >= p75 and not hub,
                   f"консолидация: плательщиков {int(row.in_deg)}, вход {_n(row.in_kzt)}",
                   f"in_deg ≥ {th.cons_min_payers}, in_kzt ≥ P75 ({_n(p75)}), не хаб"),
            _check(row.out_deg >= th.dist_min_receivers and row.fan_ratio >= th.dist_min_fan_ratio,
                   f"распределение: получателей {int(row.out_deg)}, веер {float(row.fan_ratio):.2f}",
                   f"out_deg ≥ {th.dist_min_receivers}, fan_ratio ≥ {th.dist_min_fan_ratio}"),
            _check((not bool(row.is_seed)) and row.in_deg > 0 and row.out_deg > 0
                   and th.transit_pass_lo <= pt <= th.transit_pass_hi
                   and row.in_deg <= th.transit_max_degree and row.out_deg <= th.transit_max_degree
                   and row.in_kzt >= th.transit_min_in_kzt,
                   f"транзит: {int(row.in_deg)}→{int(row.out_deg)}, out/in={pt:.2f}",
                   "все условия transit"),
        ]
        lines.append("Ни одно ролевое правило не выполнено полностью → peripheral.")

    lines += ["", f"Evidence: {row.evidence}", "Вывод является гипотезой для проверки, а не утверждением о виновности."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gid", required=True, help="18-значный идентификатор клиента")
    ap.add_argument("--out", default="out", help="папка с результатами run.py")
    ap.add_argument("--config", default=None, help="JSON с переопределением порогов")
    args = ap.parse_args()

    path = Path(args.out) / "nodes_roles.csv"
    if not path.exists():
        sys.exit(f"Нет {path} — сначала запустите: python run.py --data data --out {args.out}")
    df = pd.read_csv(path, dtype={"gid": "string"})
    hit = df.loc[df.gid == str(args.gid)]
    if hit.empty:
        sys.exit(f"gid {args.gid} не найден в {path}; всего узлов: {len(df)}")
    print(explain_row(hit.iloc[0], df, Thresholds.load(args.config)))


if __name__ == "__main__":
    main()
