"""
Правила ролей. Порядок применения фиксирован, срабатывает первое подходящее правило.
Каждая роль → формальное условие на метрики + role_score (насколько уверенно превышен порог)
+ evidence (человекочитаемое обоснование с числами, ≤ 200 символов).

Проход 1 (базовые роли):  consolidator → distributor → transit → terminal → peripheral
Проход 2 (надстройка):     coordinator — поверх базовых ролей, итеративно (до стабилизации),
                           потому что «координатор координаторов» виден только после первого круга.
"""
import numpy as np
import pandas as pd

from .config import Thresholds
from .features import fmt_kzt, pct_pos

# «структурные» плательщики для признака «консолидация второго уровня»
STRUCT_PAYERS = {"consolidator", "coordinator"}


def _clip(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))


def base_rules(df: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    recv = df.in_kzt[df.in_kzt > 0]
    cons_in_floor = recv.quantile(th.cons_min_in_pct) if len(recv) else 0
    df["p_in_kzt"] = pct_pos(df.in_kzt)
    df["p_out_kzt"] = pct_pos(df.out_kzt)

    is_cons = (df.in_deg >= th.cons_min_payers) & (df.in_kzt >= cons_in_floor)
    is_dist = (df.out_deg >= th.dist_min_receivers) & (df.fan_ratio >= th.dist_min_fan_ratio)
    pt = df.pass_through
    is_transit = (~df.is_seed) & (df.in_deg > 0) & (df.out_deg > 0) \
        & pt.between(th.transit_pass_lo, th.transit_pass_hi) \
        & (df.in_deg <= th.transit_max_degree) & (df.out_deg <= th.transit_max_degree) \
        & (df.in_kzt >= th.transit_min_in_kzt)
    big_enough = (df.in_kzt >= th.terminal_min_in_kzt) | (df.in_deg >= 2)
    observed_sink = (df.out_deg == 0) & (df.depth < 4) & (df.in_deg > 0) & big_enough
    est_sink = df.truncated_by_depth & (df.p_continue < th.terminal_max_p_continue) & big_enough

    role = np.select(
        [is_cons, is_dist, is_transit, observed_sink | est_sink],
        ["consolidator", "distributor", "transit", "terminal"],
        default="peripheral",
    )
    df["role"] = role
    df["terminal_kind"] = np.where(df.role == "terminal",
                                   np.where(observed_sink, "observed", "estimated"), "")
    return df


def coordinator_pass(G, df: pd.DataFrame, th: Thresholds, max_rounds: int = 4):
    """Кандидат в организаторы: к узлу сходятся деньги ≥ N seed, крупный вход, и он «управляет» структурой."""
    in_floor = df.in_kzt[df.in_kzt > 0].quantile(th.coord_min_in_pct) if (df.in_kzt > 0).any() else 0
    df["coord_criteria"] = ""
    for _ in range(max_rounds):
        role = dict(zip(df.gid, df.role))
        changed = False
        for i, r in df.iterrows():
            if r.seed_reach < th.coord_min_seed_reach or r.in_kzt < in_floor or r.role == "coordinator":
                continue
            # «жёлоб»: всё полученное уходит одному получателю — управляющее звено ниже по потоку
            if r.out_deg == 1 and th.transit_pass_lo <= (r.pass_through or 0) <= th.transit_pass_hi:
                continue
            v = r.gid
            struct_payers = sum(1 for u in G.predecessors(v) if role.get(u) in STRUCT_PAYERS)
            ctrl = [w for w in G.successors(v) if role.get(w) == "distributor"]
            crit = []
            if struct_payers >= th.coord_min_struct_payers:
                crit.append(f"получает от {struct_payers} консолидаторов/координаторов")
            if ctrl:
                crit.append(f"финансирует {len(ctrl)} распределителя(ей)")
            if r.n_agg_payers >= 2 and r.merge_gain >= th.coord_min_seed_reach:
                crit.append(f"сводит {r.n_agg_payers} ветви, +{r.merge_gain} seed сверх любой из них")
            if r.in_deg >= th.cons_min_payers and r.out_deg >= th.dist_min_receivers:
                crit.append("собирает и рассылает одновременно")
            if crit:
                df.at[i, "role"] = "coordinator"
                df.at[i, "coord_criteria"] = "; ".join(crit)
                changed = True
        if not changed:
            break
    return df


def scores(df: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    s = np.zeros(len(df))
    for k, r in enumerate(df.itertuples(index=False)):
        if r.role == "consolidator":
            v = 0.5 * _clip(r.in_deg / (2 * th.cons_min_payers)) + 0.3 * r.p_in_kzt + 0.2 * _clip(r.n_seed_payers / 3)
        elif r.role == "distributor":
            v = 0.5 * _clip(r.out_deg / (3 * th.dist_min_receivers)) + 0.3 * _clip(r.fan_ratio / 10) + 0.2 * r.p_out_kzt
        elif r.role == "transit":
            v = 0.5 * _clip(1 - abs(r.pass_through - 1) / 0.2) + 0.3 * float(r.fast_out_share) + 0.2 * r.p_in_kzt
        elif r.role == "terminal":
            v = (0.6 + 0.4 * r.p_in_kzt) if r.terminal_kind == "observed" else 0.6 * (1 - r.p_continue)
        elif r.role == "coordinator":
            n_crit = len([c for c in r.coord_criteria.split(";") if c.strip()])
            v = 0.4 * _clip(r.seed_reach / 15) + 0.3 * r.p_in_kzt + 0.3 * _clip(n_crit / 2)
            if r.truncated_by_depth:
                v *= 0.85       # что делает узел дальше — не видно
        else:  # peripheral: уверенность в ОТСУТСТВИИ признаков
            signal = max(r.in_deg / th.cons_min_payers, r.out_deg / th.dist_min_receivers, r.p_in_kzt)
            v = _clip(1 - signal, 0.05, 0.95)
            if r.truncated_by_depth:
                v *= 0.6        # на границе выгрузки мы знаем меньше
        s[k] = round(_clip(v), 3)
    df["role_score"] = s
    return df


def _flags(r) -> str:
    f = []
    if r.fast_out_share >= 0.7 and r.out_deg > 0 and r.in_deg > 0:
        f.append("сквозной пропуск ≤2дн")
    if r.flag_structuring:
        f.append(f"дробление у порога: {r.near_threshold_tx} перев.")
    if r.max_payers_same_day >= 3:
        f.append(f"{int(r.max_payers_same_day)} плательщ. в 1 день")
    if getattr(r, "n_cycles", 0) > 0:
        f.append("возвратный поток")
    if r.flag_anomaly:
        f.append("аномалия для колена")
    return "; ".join(f)


def evidence(df: pd.DataFrame) -> pd.DataFrame:
    ev = []
    for r in df.itertuples(index=False):
        pt = "" if pd.isna(r.pass_through) else f"{min(r.pass_through, 99) * 100:.0f}%"
        seed_note = " (seed: входящие занижены)" if r.is_seed else ""
        if r.role == "consolidator":
            t = (f"Получает от {r.in_deg} плательщиков ({r.n_seed_payers} seed) {fmt_kzt(r.in_kzt)}; "
                 f"передаёт дальше {pt or '0%'}; сходятся потоки {r.seed_reach} seed{seed_note}")
        elif r.role == "distributor":
            t = (f"Рассылает на {r.out_deg} получателей {fmt_kzt(r.out_kzt)} (веер {r.fan_ratio:.0f}:1); "
                 f"плательщиков {r.in_deg}{seed_note}")
        elif r.role == "transit":
            t = (f"Пропуск {pt}: получил {fmt_kzt(r.in_kzt)}, отдал {fmt_kzt(r.out_kzt)}; "
                 f"{r.in_deg}→{r.out_deg} контрагента; в ≤2 дн ушло {r.fast_out_share * 100:.0f}%")
        elif r.role == "terminal" and r.terminal_kind == "observed":
            t = (f"Получил {fmt_kzt(r.in_kzt)} от {r.in_deg} плательщ.; исходящих нет, "
                 f"хотя колено {r.depth} обходилось — деньги остались")
        elif r.role == "terminal":
            t = (f"Колено 4, исходящие не выгружались; модель по 1–3 коленам: P(ушли дальше)="
                 f"{r.p_continue:.2f}; получил {fmt_kzt(r.in_kzt)}")
        elif r.role == "coordinator":
            t = f"Сходятся деньги {r.seed_reach} seed, вход {fmt_kzt(r.in_kzt)}; {r.coord_criteria}"
            if r.truncated_by_depth:
                t += "; колено 4 — исходящие не видны"
        else:
            if r.in_deg == 0 and r.out_deg == 0:
                t = "Нет переводов ≥5 000 KZT внутри банка в выгрузке; признаков роли нет"
            elif r.truncated_by_depth and not pd.isna(r.p_continue):
                t = (f"Обрыв выгрузки (колено 4): получил {fmt_kzt(r.in_kzt)} от {r.in_deg}; "
                     f"P(ушли дальше)={r.p_continue:.2f} — нужен 5-й уровень")
            else:
                t = (f"Вход {fmt_kzt(r.in_kzt)} от {r.in_deg}, выход {fmt_kzt(r.out_kzt)} на {r.out_deg}; "
                     f"пороги ролей не достигнуты{seed_note}")
        fl = _flags(r)
        if fl and len(t) + len(fl) + 3 <= 200:
            t = f"{t} [{fl}]"
        ev.append(t[:200])
    df["evidence"] = ev
    return df


def assign_roles(G, df: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    df = base_rules(df, th)
    df = coordinator_pass(G, df, th)
    df = scores(df, th)
    df = evidence(df)
    return df
