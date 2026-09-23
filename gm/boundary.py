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

Модель — логистическая регрессия, коэффициенты печатаются в отчёт (не чёрный ящик).
Качество проверяется двумя способами: обычная 5-fold CV и «сдвиг по колену» (учим на 1–2,
проверяем на 3) — это ближе к реальному переносу на 4-е колено. Если CV AUC < MIN_AUC,
прогноз не делается: всем обрезанным ставится базовая доля, и это пишется в отчёт.
Сумма (1 − P) по обрезанным — оценка числа настоящих «стоков».
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MIN_AUC = 0.6

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


def _auc_shift(X, y, depth):
    """Учим на коленах 1–2, проверяем на 3-м: проверка переноса на более глубокое колено."""
    tr, te = depth.isin([1, 2]).to_numpy(), (depth == 3).to_numpy()
    if y[tr].nunique() < 2 or y[te].nunique() < 2:
        return None
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
    return float(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))


def fit_boundary_model(df: pd.DataFrame, tx: pd.DataFrame):
    X_all = forward_features(df, tx)
    train_mask = ((~df.is_seed) & df.depth.between(1, 3) & (X_all.in_deg_fwd > 0)).to_numpy()
    X, y = X_all[train_mask].reset_index(drop=True), (df.out_deg[train_mask] > 0).astype(int).reset_index(drop=True)
    base = float(y.mean()) if len(y) else None
    report = {"n_train": int(len(y)), "base_rate_has_out": base,
              "features": "только переводы от более мелких колён + контекст плательщиков (без утечки)"}
    df["p_continue"] = np.nan
    mask = (df.truncated_by_depth & (X_all.in_deg_fwd > 0)).to_numpy()
    fallback = 0.5 if base is None else base            # 0.0 — валидная доля, не заменяем её на 0.5
    if len(y) < 50 or y.nunique() < 2:
        report["status"] = "мало данных — используем долю из 1–3 колена как константу"
        df.loc[mask, "p_continue"] = fallback
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        auc = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
        model.fit(X, y)
        report.update({
            "status": "ok",
            "cv_auc_mean": float(auc.mean()),
            "shift_auc_depth12_to_3": _auc_shift(X, y, df.depth[train_mask].reset_index(drop=True)),
            "coefficients_std": {k: round(float(c), 3) for k, c in zip(FEATURES, model[-1].coef_[0])},
        })
        if report["cv_auc_mean"] < MIN_AUC:
            # честно: признаки не различают «ушли дальше» и «осели» → не делаем вид, что знаем
            report["status"] = f"слабая модель (AUC<{MIN_AUC}) — для всех обрезанных берём базовую долю"
            df.loc[mask, "p_continue"] = fallback
        elif mask.any():
            df.loc[mask, "p_continue"] = model.predict_proba(X_all[mask])[:, 1]
    report["n_truncated"] = int(df.truncated_by_depth.sum())
    report["expected_true_terminals"] = float((1 - df.loc[mask, "p_continue"]).sum())
    return df, report
