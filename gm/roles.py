"""
Правила ролей. Порядок применения фиксирован, срабатывает первое подходящее правило.
Каждая роль → формальное условие на метрики + role_score (насколько уверенно превышен порог)
+ evidence (человекочитаемое обоснование с числами, ≤ 200 символов).

Проход 1 (базовые роли):  consolidator → distributor → transit → terminal → peripheral
Проход 2 (надстройка):     coordinator — поверх базовых ролей, итеративно (до стабилизации),
                           потому что «координатор координаторов» виден только после первого круга.

Словарь ролей фиксирован ТЗ (6 ролей). Типологии, которые в него не входят (дроп, обменник,
обнал, внешнее финансирование, розничный сбор), выносятся в отдельную колонку pattern.
"""
import numpy as np
import pandas as pd

from .config import Thresholds
from .features import fmt_kzt, pct_pos

# «структурные» плательщики для признака «консолидация второго уровня»
STRUCT_PAYERS = {"consolidator", "coordinator"}
SOLO_CRIT = "собирает и рассылает одновременно"

PATTERN_RU = {
    "drop_like": "дроп: быстрый сквозной пропуск 1–2 получателям",
    "exchanger_like": "похож на P2P-обменник/хаб: много контрагентов в обе стороны, пропуск ≈100%",
    "cash_out_candidate": "кандидат на вывод: деньги seed пришли и не ушли внутри банка",
    "external_funding": "источник средств вне выборки: отдал заметно больше видимого входа",
    "retail_collection": "розничный сбор: много мелких поступлений",
}


def _clip(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))


def base_rules(df: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    recv = df.in_kzt[df.in_kzt > 0]
    cons_in_floor = recv.quantile(th.cons_min_in_pct) if len(recv) else 0
    df["p_in_kzt"] = pct_pos(df.in_kzt)
    df["p_out_kzt"] = pct_pos(df.out_kzt)
    pt = df.pass_through

    # консолидатор собирает: если он отдал в разы больше, чем получил, и сам рассылает веером —
    # это хаб с внешним финансированием или распределитель, а не сборщик
    hub_like = (pt > th.cons_hub_pass) & (df.out_deg >= th.dist_min_receivers)
    is_cons = (df.in_deg >= th.cons_min_payers) & (df.in_kzt >= cons_in_floor) & ~hub_like
    is_dist = (df.out_deg >= th.dist_min_receivers) & (df.fan_ratio >= th.dist_min_fan_ratio)
    is_transit = (~df.is_seed) & (df.in_deg > 0) & (df.out_deg > 0) \
        & pt.between(th.transit_pass_lo, th.transit_pass_hi) \
        & (df.in_deg <= th.transit_max_degree) & (df.out_deg <= th.transit_max_degree) \
        & (df.in_kzt >= th.transit_min_in_kzt)
    big_enough = (df.in_kzt >= th.terminal_min_in_kzt) | (df.in_deg >= 2)
    observed_sink = (df.out_deg == 0) & (df.depth < 4) & (df.in_deg > 0) & big_enough
    # частичный сток: исходящие видны, но дальше ушла малая доля крупного входа
    partial_sink = (~df.is_seed) & (df.depth < 4) & (df.out_deg > 0) \
        & (pt <= th.terminal_max_pass) & (df.in_kzt >= th.terminal_min_partial_kzt)
    est_sink = df.truncated_by_depth & (df.p_continue < th.terminal_max_p_continue) & big_enough

    role = np.select(
        [is_cons, is_dist, is_transit, observed_sink | partial_sink | est_sink],
        ["consolidator", "distributor", "transit", "terminal"],
        default="peripheral",
    )
    df["role"] = role
    kind = np.select([observed_sink, partial_sink, est_sink], ["observed", "partial", "estimated"], default="none")
    df["terminal_kind"] = np.where(df.role == "terminal", kind, "none")
    return df


def coordinator_pass(G, df: pd.DataFrame, th: Thresholds, max_rounds: int = 4):
    """Кандидат в верхний уровень: по хронологии к узлу доходят деньги ≥ N seed, крупный вход,
    и он «управляет» структурой. «Собирает и рассылает одновременно» само по себе НЕ основание —
    так же выглядят P2P-обменники и хабы (они помечаются pattern=exchanger_like)."""
    in_floor = df.in_kzt[df.in_kzt > 0].quantile(th.coord_min_in_pct) if (df.in_kzt > 0).any() else 0
    df["coord_criteria"] = "none"
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
            if r.n_agg_payers >= 2 and r.merge_gain >= th.coord_min_merge_gain:
                crit.append(f"сводит {r.n_agg_payers} ветви, +{r.merge_gain} seed сверх любой из них")
            if r.in_deg >= th.cons_min_payers and r.out_deg >= th.dist_min_receivers:
                crit.append(SOLO_CRIT)
            if crit and crit != [SOLO_CRIT]:
                df.at[i, "role"] = "coordinator"
                df.at[i, "terminal_kind"] = "none"
                df.at[i, "coord_criteria"] = "; ".join(crit)
                changed = True
        if not changed:
            break
    return df


def patterns(df: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    """Типологии вне словаря ролей ТЗ — отдельные теги, через «;». «none», если признаков нет."""
    pt = df.pass_through.fillna(0)
    tags = {
        "drop_like": (~df.is_seed) & (df.fast_out_share >= 0.7) & df.out_deg.between(1, 2)
        & pt.between(th.transit_pass_lo, th.transit_pass_hi) & (df.in_kzt >= th.transit_min_in_kzt),
        "exchanger_like": (df.in_deg >= th.cons_min_payers) & (df.out_deg >= th.cons_min_payers)
        & pt.between(0.8, 1.25) & (df.active_in_days >= 10),
        "cash_out_candidate": (df.role == "terminal") & df.terminal_kind.isin(["observed", "partial"])
        & (df.n_seed_payers >= 1),
        "external_funding": (~df.is_seed) & (df.out_kzt >= th.terminal_min_in_kzt)
        & ((pt >= th.external_pass) | (df.out_before_any_in_kzt >= 0.5 * df.out_kzt)),
        "retail_collection": df.flag_structuring
        | ((df.in_tx >= th.small_ticket_min_tx) & (df.mean_in_ticket.fillna(np.inf) <= th.small_ticket_max)),
    }
    out = pd.Series("", index=df.index)
    for name, m in tags.items():
        out = np.where(m, np.where(out == "", name, out + ";" + name), out)
    df["pattern"] = pd.Series(out, index=df.index).replace("", "none")
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
            if r.terminal_kind == "observed":
                v = 0.6 + 0.4 * r.p_in_kzt
            elif r.terminal_kind == "partial":
                v = 0.5 + 0.3 * _clip(1 - r.pass_through / th.terminal_max_pass) + 0.2 * r.p_in_kzt
            else:
                v = 0.6 * (1 - r.p_continue)
        elif r.role == "coordinator":
            n_crit = len([c for c in r.coord_criteria.split(";") if c.strip()])
            v = 0.4 * _clip(r.seed_reach / 7) + 0.3 * r.p_in_kzt + 0.3 * _clip(n_crit / 2)
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


def _flags(r, th) -> str:
    f = []
    if r.fast_out_share >= 0.7 and r.out_deg > 0 and r.in_deg > 0:
        f.append(f"≤{th.fast_days}дн ушло {r.fast_out_share * 100:.0f}%")
    if r.flag_structuring:
        f.append(f"{r.near_threshold_tx} мелк. поступл. {th.structuring_lo / 1e3:.0f}–{th.structuring_hi / 1e3:.0f} тыс.")
    if r.max_payers_same_day >= 3:
        f.append(f"{int(r.max_payers_same_day)} плательщ. в 1 день")
    if getattr(r, "n_cycles", 0) > 0:
        f.append(f"в {r.n_cycles} циклах")
    if r.flag_anomaly:
        f.append(f"нетипично для колена: {r.anomaly_feature} z={r.anomaly_z:.0f}")
    return "; ".join(f)


def _tx(n, kzt):
    """Ловушка №3: одна сумма может быть одним переводом или сорока — показываем оба сигнала."""
    return f"{n} перев., ср. {fmt_kzt(kzt / n)}" if n else "0 перев."


def _nearest_miss(r, th) -> str:
    """Для периферии: какой порог ближе всего и насколько не дотянул — чтобы роль можно было защитить."""
    cand = [
        (r.in_deg / th.cons_min_payers, f"плательщиков {r.in_deg}<{th.cons_min_payers} для консолидатора"),
        (r.out_deg / th.dist_min_receivers, f"получателей {r.out_deg}<{th.dist_min_receivers} для распределителя"),
    ]
    pt = r.pass_through
    if r.in_deg and r.out_deg and not pd.isna(pt):
        if pt < th.transit_pass_lo:
            cand.append((pt / th.transit_pass_lo, f"пропуск {pt * 100:.0f}%<{th.transit_pass_lo * 100:.0f}% для транзита"))
        elif pt > th.transit_pass_hi:
            cand.append((th.transit_pass_hi / pt, f"пропуск {pt * 100:.0f}%>{th.transit_pass_hi * 100:.0f}% для транзита"))
    ratio, text = max(cand, key=lambda x: x[0])
    return text


def evidence(df: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    ev = []
    for r in df.itertuples(index=False):
        pt = r.pass_through
        seed_note = " (seed: вход занижен)" if r.is_seed else ""
        external = (not pd.isna(pt)) and pt >= th.external_pass
        if r.role == "consolidator":
            t = (f"Получает от {r.in_deg} плательщ. ({r.n_seed_payers} seed) {fmt_kzt(r.in_kzt)}, {_tx(r.in_tx, r.in_kzt)}; "
                 f"по хронологии доходят деньги {r.seed_reach} seed{seed_note}")
        elif r.role == "distributor":
            src = (f"отдал в {pt:.0f}× больше видимого входа" if external else f"плательщиков {r.in_deg}")
            t = (f"Рассылает {r.out_deg} получателям {fmt_kzt(r.out_kzt)} (веер {r.fan_ratio:.0f}:1), "
                 f"{_tx(r.out_tx, r.out_kzt)}; {src}{seed_note}")
        elif r.role == "transit":
            t = (f"Пропуск {pt * 100:.0f}%: получил {fmt_kzt(r.in_kzt)} ({_tx(r.in_tx, r.in_kzt)}), "
                 f"отдал {fmt_kzt(r.out_kzt)}; {r.in_deg}→{r.out_deg} контрагента")
        elif r.role == "terminal" and r.terminal_kind == "observed":
            t = (f"Получил {fmt_kzt(r.in_kzt)} от {r.in_deg} плательщ. ({_tx(r.in_tx, r.in_kzt)}); "
                 f"внутрибанк. исходящих ≥5 тыс. нет на колене {r.depth}")
        elif r.role == "terminal" and r.terminal_kind == "partial":
            t = (f"Получил {fmt_kzt(r.in_kzt)} от {r.in_deg} ({_tx(r.in_tx, r.in_kzt)}), дальше ушло "
                 f"{pt * 100:.0f}% — осело {fmt_kzt(r.in_kzt - r.out_kzt)} внутри банка")
        elif r.role == "terminal":
            t = (f"Колено 4, исходящие не выгружались; модель по 1–3 коленам: P(ушли дальше)="
                 f"{r.p_continue:.2f}; получил {fmt_kzt(r.in_kzt)}")
        elif r.role == "coordinator":
            t = f"По хронологии доходят деньги {r.seed_reach} seed, вход {fmt_kzt(r.in_kzt)}; {r.coord_criteria}"
            if r.truncated_by_depth:
                t += "; колено 4 — исходящие не видны"
        else:
            if r.in_deg == 0 and r.out_deg == 0:
                t = "Нет переводов ≥5 000 KZT внутри банка в выгрузке; признаков роли нет"
            elif r.truncated_by_depth and not pd.isna(r.p_continue) and r.p_continue >= 0:
                t = (f"Обрыв выгрузки (колено 4): получил {fmt_kzt(r.in_kzt)} от {r.in_deg}; "
                     f"P(ушли дальше)={r.p_continue:.2f} — нужен 5-й уровень")
            else:
                t = (f"Вход {fmt_kzt(r.in_kzt)} от {r.in_deg}, выход {fmt_kzt(r.out_kzt)} на {r.out_deg}; "
                     f"роль не выражена: {_nearest_miss(r, th)}{seed_note}")
        fl = _flags(r, th)
        if fl and len(t) + len(fl) + 3 <= 200:
            t = f"{t} [{fl}]"
        ev.append(t[:200])
    df["evidence"] = ev
    return df


def assign_roles(G, df: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    df = base_rules(df, th)
    df = coordinator_pass(G, df, th)
    df = scores(df, th)
    df = patterns(df, th)
    df = evidence(df, th)
    return df
