"""
Артефакт обрыва на 4-м колене.

Проблема: у узла 4-го колена исходящие не выгружались. out_deg=0 там ничего не значит.

Идея: у узлов 1–3 колена исходящие видны полностью (в пределах банка и порога 5 000),
значит на них можно выучить, КАК по профилю входящих отличить «деньги ушли дальше»
от «деньги остались», и применить к 4-му колену.

Защита от утечки. У узла 4-го колена видны ТОЛЬКО входящие от более мелких колён (3→4):
переводы 4→4 и 5→4 не выгружались. У узлов 1–3 бывают входящие от своего или более глубокого
колена (обратные потоки, циклы) — и именно они сильно связаны с наличием исходящих. Если учить
на всех входящих, модель выучит этот артефакт выгрузки, а на 4-м колене его нет по построению.
Поэтому все признаки считаются только по «прямым» переводам depth(плательщик) < depth(получатель),
плюс контекст плательщиков (их исходящие видны полностью — они на коленах 0–3).

Модель — логистическая регрессия, коэффициенты с bootstrap-интервалами печатаются в отчёт.
Качество: 5-fold CV (стратифицированная, перемешанная, seed фиксирован) и «сдвиг по колену»
(учим на 1–2, проверяем на 3) — это ближе к реальному переносу на 4-е колено.

Калибровка. p_continue сравнивается с порогом terminal_max_p_continue, поэтому вероятность должна
значить то, что написано («0.35» ≈ 35% узлов с таким p действительно имеют исходящие).
Проверяется на out-of-fold предсказаниях (reliability diagram, Brier, ECE). Калибратор:
Platt (sigmoid) по умолчанию; isotonic — только если снижает Brier ≥ 1% (на ~1.7 тыс. примеров
isotonic легко переобучается ступеньками).

Деградация. Если CV AUC < MIN_AUC, прогноз не делается: всем обрезанным ставится базовая доля
продолжения из 1–3 колена. Если она выше порога (обычно так: ~0.37 > 0.35), ни один узел 4-го
колена не получит terminal(estimated) — система честно «не знает», а не угадывает.
"""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MIN_AUC = 0.6
CV_SEED = 0
N_BOOT = 200
ISOTONIC_MIN_GAIN = 0.01          # isotonic выбирается, только если Brier ниже на ≥ 1%
THRESHOLD_GRID = [round(x, 2) for x in np.arange(0.15, 0.61, 0.05)]
MIN_PRECISION_LIFT = 0.10         # правило выбора порога: precision ≥ доля стоков + 0.10

FEATURES = {
    "log_in_kzt_fwd": "лог. входящий оборот (от более мелких колён)",
    "in_deg_fwd": "число плательщиков",
    "log_in_tx_fwd": "лог. число входящих переводов",
    "log_mean_ticket_fwd": "лог. средний входящий чек",
    "active_in_days_fwd": "дней с поступлениями",
    "days_left_fwd": "дней до конца периода после последнего поступления",
    "payer_outdeg_mean": "среднее число получателей у плательщиков",
    "max_share_of_payer_out": "макс. доля исходящих плательщика, пришедшая узлу",
}


def forward_features(df: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    """Признаки только по переводам depth(src) < depth(dst) — то, что видно и у 4-го колена."""
    depth = dict(zip(df.gid, df.depth))
    out_deg = dict(zip(df.gid, df.out_deg))
    out_kzt = dict(zip(df.gid, df.out_kzt))
    t = tx[["src", "dst", "date", "sum_kzt"]].copy()
    t["day"] = pd.to_datetime(t.date).dt.normalize()
    last_day = t.day.max()
    fwd = t[t.src.map(depth) < t.dst.map(depth)]
    g = fwd.groupby("dst")
    pair = fwd.groupby(["dst", "src"]).sum_kzt.sum().reset_index()
    pair["payer_out_deg"] = pair.src.map(out_deg)
    pair["share"] = pair.sum_kzt / pair.src.map(out_kzt).replace(0, np.nan)
    pg = pair.groupby("dst")
    f = pd.DataFrame({
        "log_in_kzt_fwd": np.log1p(g.sum_kzt.sum()),
        "in_deg_fwd": g.src.nunique(),
        "log_in_tx_fwd": np.log1p(g.size()),
        "log_mean_ticket_fwd": np.log1p(g.sum_kzt.mean()),
        "active_in_days_fwd": g.day.nunique(),
        "days_left_fwd": (last_day - g.day.max()).dt.days,
        "payer_outdeg_mean": pg.payer_out_deg.mean(),
        "max_share_of_payer_out": pg.share.max(),
    })
    return f.reindex(df.gid).fillna(0).reset_index(drop=True)


def _base():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def _auc_shift(X, y, depth):
    """Учим на коленах 1–2, проверяем на 3-м: проверка переноса на более глубокое колено."""
    tr, te = depth.isin([1, 2]).to_numpy(), (depth == 3).to_numpy()
    if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
        return None
    m = _base().fit(X[tr], y[tr])
    return float(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))


def _reliability(y, p, n_bins=10):
    """Reliability diagram по квантильным корзинам + ECE (взвешенное |факт − прогноз|)."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    bins, ece = [], 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        pm, fr = float(p[m].mean()), float(y[m].mean())
        bins.append({"p_mean": round(pm, 4), "frac_continue": round(fr, 4), "n": int(m.sum())})
        ece += m.mean() * abs(fr - pm)
    return bins, float(ece)


def _calibrate(X, y, cv):
    """Сравнить калибраторы на out-of-fold предсказаниях; вернуть лучший метод и OOF-вероятности."""
    oof = {"raw": cross_val_predict(_base(), X, y, cv=cv, method="predict_proba")[:, 1]}
    for m in ("sigmoid", "isotonic"):
        oof[m] = cross_val_predict(CalibratedClassifierCV(_base(), method=m, cv=5), X, y,
                                   cv=cv, method="predict_proba")[:, 1]
    brier = {k: float(brier_score_loss(y, v)) for k, v in oof.items()}
    method = "isotonic" if brier["isotonic"] <= brier["sigmoid"] * (1 - ISOTONIC_MIN_GAIN) else "sigmoid"
    bins, ece = _reliability(y, oof[method])
    bins_raw, ece_raw = _reliability(y, oof["raw"])
    return method, oof[method], {
        "method": method,
        "rule": f"isotonic только при снижении Brier ≥ {ISOTONIC_MIN_GAIN:.0%}, иначе Platt (sigmoid)",
        "brier": {k: round(v, 4) for k, v in brier.items()},
        "ece": round(ece, 4), "ece_raw": round(ece_raw, 4),
        "reliability": bins, "reliability_raw": bins_raw,
        "oof_auc": round(float(roc_auc_score(y, oof[method])), 4),
    }


def _bootstrap_coefs(X, y, n=N_BOOT, seed=CV_SEED):
    """95% bootstrap-интервалы стандартизованных коэффициентов и доля выборок с тем же знаком."""
    rng = np.random.default_rng(seed)
    full = _base().fit(X, y)[-1].coef_[0]
    B = []
    Xn, yn = X.to_numpy(), np.asarray(y)
    while len(B) < n:
        i = rng.integers(0, len(yn), len(yn))
        if len(set(yn[i])) < 2:
            continue
        B.append(_base().fit(Xn[i], yn[i])[-1].coef_[0])
    B = np.array(B)
    out = {}
    for j, k in enumerate(FEATURES):
        lo, hi = np.percentile(B[:, j], [2.5, 97.5])
        out[k] = {"coef": round(float(full[j]), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
                  "sign_stable": round(float((np.sign(B[:, j]) == np.sign(full[j])).mean()), 3),
                  "significant": bool(lo > 0 or hi < 0), "meaning": FEATURES[k]}
    return out


def _threshold_curve(y, p, p_trunc):
    """Для каждого порога: что будет, если считать узел стоком при p_continue < порога.
    precision/recall — на OOF-предсказаниях 1–3 колена (там правда известна);
    n_depth4 и expected_precision_depth4 = mean(1 − p) — на реальных узлах 4-го колена."""
    sink = np.asarray(y) == 0
    rows = []
    for t in THRESHOLD_GRID:
        s = p < t
        s4 = p_trunc < t
        rows.append({
            "threshold": t, "n_selected_cv": int(s.sum()),
            "precision_cv": round(float(sink[s].mean()), 4) if s.any() else None,
            "recall_cv": round(float((s & sink).sum() / max(sink.sum(), 1)), 4),
            "n_depth4": int(s4.sum()),
            "expected_precision_depth4": round(float((1 - p_trunc[s4]).mean()), 4) if s4.any() else None,
        })
    return rows


def recommend_threshold(curve, base_sink_rate, lift=MIN_PRECISION_LIFT):
    """Правило: наибольший порог (больше recall), при котором precision на CV ≥ доля стоков + lift.
    Ниже — узлы 4-го колена почти не отбираются; выше — точность падает к уровню «пометить всех»."""
    need = base_sink_rate + lift
    ok = [r for r in curve if r["precision_cv"] is not None and r["precision_cv"] >= need and r["n_depth4"] > 0]
    return (max(ok, key=lambda r: r["threshold"])["threshold"] if ok else None), round(need, 4)


def threshold_summary(report: dict, threshold: float) -> dict:
    """Строка кривой для порога из config + рекомендованный правилом порог (для отчёта и README)."""
    curve = report.get("threshold_curve") or []
    row = min(curve, key=lambda r: abs(r["threshold"] - threshold)) if curve else None
    return {"configured": threshold, "recommended": report.get("recommended_threshold"),
            "rule": f"наибольший порог с precision_cv ≥ доля стоков + {MIN_PRECISION_LIFT}",
            "required_precision": report.get("required_precision"), "at_configured": row,
            "agrees": report.get("recommended_threshold") == threshold}


def fit_boundary_model(df: pd.DataFrame, tx: pd.DataFrame):
    X_all = forward_features(df, tx)
    train_mask = ((~df.is_seed) & df.depth.between(1, 3) & (X_all.in_deg_fwd > 0)).to_numpy()
    X, y = X_all[train_mask].reset_index(drop=True), (df.out_deg[train_mask] > 0).astype(int).to_numpy()
    base = float(y.mean()) if len(y) else None
    report = {"n_train": int(len(y)), "base_rate_has_out": base,
              "features": "только переводы от более мелких колён + контекст плательщиков (без утечки)",
              "min_auc": MIN_AUC}
    df["p_continue"] = np.nan
    mask = (df.truncated_by_depth & (X_all.in_deg_fwd > 0)).to_numpy()
    fallback = 0.5 if base is None else base            # 0.0 — валидная доля, не заменяем её на 0.5
    if len(y) < 50 or len(set(y)) < 2 or min(np.bincount(y)) < 25:   # вложенная CV 5×5 требует оба класса в каждом фолде
        report["status"] = "мало данных — используем долю из 1–3 колена как константу"
        report["degraded_to_baseline"] = True
        df.loc[mask, "p_continue"] = fallback
    else:
        cv = StratifiedKFold(5, shuffle=True, random_state=CV_SEED)
        method, oof, calib = _calibrate(X, y, cv)
        auc_raw = float(roc_auc_score(y, cross_val_predict(_base(), X, y, cv=cv, method="predict_proba")[:, 1]))
        report.update({
            "status": "ok",
            "cv_auc_mean": auc_raw,
            "shift_auc_depth12_to_3": _auc_shift(X, y, df.depth[train_mask].reset_index(drop=True)),
            "calibration": calib,
            "feature_importance_ci": _bootstrap_coefs(X, y),
        })
        report["coefficients_std"] = {k: v["coef"] for k, v in report["feature_importance_ci"].items()}
        model = CalibratedClassifierCV(_base(), method=method, cv=5).fit(X, y)
        p_trunc = model.predict_proba(X_all[mask])[:, 1] if mask.any() else np.array([])
        report["threshold_curve"] = _threshold_curve(y, oof, p_trunc)
        report["recommended_threshold"], report["required_precision"] = recommend_threshold(
            report["threshold_curve"], 1 - base)
        if auc_raw < MIN_AUC:
            # честно: признаки не различают «ушли дальше» и «осели» → не делаем вид, что знаем
            report["status"] = f"слабая модель (AUC<{MIN_AUC}) — для всех обрезанных берём базовую долю"
            report["degraded_to_baseline"] = True
            df.loc[mask, "p_continue"] = fallback
        else:
            report["degraded_to_baseline"] = False
            if mask.any():
                df.loc[mask, "p_continue"] = p_trunc
    report["n_truncated"] = int(df.truncated_by_depth.sum())
    report["expected_true_terminals"] = float((1 - df.loc[mask, "p_continue"]).sum())
    return df, report


# ---------------------------------------------------------------- отчёт: reliability diagram
def calibration_svg(report: dict, w=360, h=300) -> str:
    """Reliability diagram (SVG без внешних библиотек): по оси X — предсказанная P(ушли дальше),
    по Y — фактическая доля узлов с исходящими. Диагональ — идеальная калибровка."""
    c = report.get("calibration")
    if not c:
        return ""
    P = 40
    X = lambda v: P + v * (w - P - 10)
    Y = lambda v: h - P - v * (h - P - 10)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
         f'font-family="sans-serif" font-size="10"><rect width="{w}" height="{h}" fill="#fff"/>']
    for t in (0, 0.25, 0.5, 0.75, 1):
        s.append(f'<line x1="{X(0)}" x2="{X(1)}" y1="{Y(t)}" y2="{Y(t)}" stroke="#e3e6ec"/>'
                 f'<text x="4" y="{Y(t) + 3}" fill="#6b7280">{t:.2f}</text>'
                 f'<text x="{X(t) - 8}" y="{h - P + 14}" fill="#6b7280">{t:.2f}</text>')
    s.append(f'<line x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(1)}" stroke="#9aa3b2" stroke-dasharray="4 3"/>')
    for key, col, lbl in (("reliability_raw", "#9aa3b2", "без калибровки"),
                          ("reliability", "#2f5bd3", f"калибровка: {c['method']}")):
        pts = " ".join(f"{X(b['p_mean']):.1f},{Y(b['frac_continue']):.1f}" for b in c[key])
        s.append(f'<polyline fill="none" stroke="{col}" stroke-width="2" points="{pts}"/>')
        s += [f'<circle cx="{X(b["p_mean"]):.1f}" cy="{Y(b["frac_continue"]):.1f}" r="3" fill="{col}"/>' for b in c[key]]
    s.append(f'<text x="{P}" y="14" fill="#2f5bd3">● калибровка {c["method"]} (ECE {c["ece"]:.3f})</text>'
             f'<text x="{P}" y="27" fill="#9aa3b2">● без калибровки (ECE {c["ece_raw"]:.3f})</text>'
             f'<text x="{w / 2 - 70}" y="{h - 6}" fill="#1d2330">предсказанная P(ушли дальше)</text>'
             f'<text x="10" y="{h / 2 + 60}" fill="#1d2330" transform="rotate(-90 10 {h / 2 + 60})">факт: доля с исходящими</text>'
             '</svg>')
    return "".join(s)


def calibration_markdown(report: dict, threshold: float, svg_name="calibration.svg") -> list:
    """Раздел report.md: калибровка, интервалы коэффициентов, кривая порогов."""
    if report.get("status") != "ok" and not report.get("calibration"):
        return ["", "## Модель обрыва 4-го колена", "", f"Статус: {report.get('status')}."]
    c = report["calibration"]
    L = ["", "## Модель обрыва 4-го колена: калибровка и порог", "",
         f"Статус: **{report['status']}**. CV ROC-AUC {report['cv_auc_mean']:.3f} "
         f"(порог отказа от прогноза {report['min_auc']}); перенос 1–2 → 3 колено: "
         f"{report['shift_auc_depth12_to_3']:.3f}. Обучено на {report['n_train']} узлах 1–3 колена, "
         f"доля с исходящими {report['base_rate_has_out']:.3f}.", "",
         f"![reliability diagram]({svg_name})", "",
         f"Калибратор: **{c['method']}** ({c['rule']}). Brier: без калибровки {c['brier']['raw']}, "
         f"sigmoid {c['brier']['sigmoid']}, isotonic {c['brier']['isotonic']}. ECE {c['ece']} "
         f"(без калибровки {c['ece_raw']}).", "",
         "| корзина | средний прогноз | факт (доля с исходящими) | узлов |", "|---|---|---|---|"]
    L += [f"| {i + 1} | {b['p_mean']:.3f} | {b['frac_continue']:.3f} | {b['n']} |" for i, b in enumerate(c["reliability"])]
    L += ["", "### Коэффициенты (стандартизованные признаки, 95% bootstrap-интервал, n=200)", "",
          "| признак | коэф. | 95% ДИ | знак устойчив | значим |", "|---|---|---|---|---|"]
    for k, v in report["feature_importance_ci"].items():
        L.append(f"| {v['meaning']} | {v['coef']:+.3f} | [{v['ci95'][0]:+.3f}; {v['ci95'][1]:+.3f}] | "
                 f"{v['sign_stable']:.0%} | {'да' if v['significant'] else 'нет'} |")
    ts = threshold_summary(report, threshold)
    L += ["", "### Выбор порога terminal_max_p_continue", "",
          f"Правило: {ts['rule']} = {ts['required_precision']}. Рекомендовано правилом: "
          f"**{ts['recommended']}**, в config: **{threshold}** ({'совпадает' if ts['agrees'] else 'НЕ совпадает'}).", "",
          "| порог | отобрано на CV | precision (CV) | recall (CV) | узлов 4-го колена | ожид. precision на 4-м |",
          "|---|---|---|---|---|---|"]
    for r in report["threshold_curve"]:
        f = lambda v: "—" if v is None else f"{v:.3f}"
        L.append(f"| {r['threshold']:.2f} | {r['n_selected_cv']} | {f(r['precision_cv'])} | {f(r['recall_cv'])} | "
                 f"{r['n_depth4']} | {f(r['expected_precision_depth4'])} |")
    return L
