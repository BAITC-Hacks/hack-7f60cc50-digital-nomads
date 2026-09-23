"""
Артефакт обрыва на 4-м колене.

Проблема: у узла 4-го колена исходящие не выгружались. out_deg=0 там ничего не значит.

Идея: у узлов 1–3 колена исходящие видны полностью (в пределах банка и порога 5 000),
значит на них можно выучить, КАК по профилю входящих отличить «деньги ушли дальше»
от «деньги остались». Модель видит только то, что у 4-го колена тоже есть:
входящие переводы, их суммы, число плательщиков и даты.

Модель — логистическая регрессия на 6 признаках: коэффициенты печатаются в отчёт,
то есть это не чёрный ящик. Для каждого обрезанного узла получаем P(продолжение):
  * P < terminal_max_p_continue  → «вероятный конечный получатель» (terminal, пониженный score)
  * иначе                         → «обрыв выгрузки, вероятно деньги ушли дальше» (peripheral + флаг)
Сумма (1 − P) по обрезанным — оценка числа настоящих «стоков» среди 444.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MIN_AUC = 0.6

FEATURES = {
    "log_in_kzt": "лог. входящий оборот",
    "in_deg": "число плательщиков",
    "log_in_tx": "лог. число входящих переводов",
    "log_mean_ticket": "лог. средний входящий чек",
    "active_in_days": "дней с поступлениями",
    "days_left_after_last_in": "дней до конца периода после последнего поступления",
}


def _X(df):
    return pd.DataFrame({
        "log_in_kzt": np.log1p(df.in_kzt),
        "in_deg": df.in_deg,
        "log_in_tx": np.log1p(df.in_tx),
        "log_mean_ticket": np.log1p(df.mean_in_ticket.fillna(0)),
        "active_in_days": df.active_in_days,
        "days_left_after_last_in": df.days_left_after_last_in.fillna(0),
    })


def fit_boundary_model(df: pd.DataFrame):
    train = df[(~df.is_seed) & df.depth.between(1, 3) & (df.in_deg > 0)]
    y = (train.out_deg > 0).astype(int)
    report = {"n_train": int(len(train)), "base_rate_has_out": float(y.mean()) if len(y) else None}
    df["p_continue"] = np.nan
    if len(train) < 50 or y.nunique() < 2:
        report["status"] = "мало данных — используем долю из 1–3 колена как константу"
        df.loc[df.truncated_by_depth, "p_continue"] = report["base_rate_has_out"] or 0.5
        return df, report
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    auc = cross_val_score(model, _X(train), y, cv=5, scoring="roc_auc")
    model.fit(_X(train), y)
    coefs = model[-1].coef_[0]
    report.update({
        "status": "ok",
        "cv_auc_mean": float(auc.mean()),
        "coefficients_std": {k: round(float(c), 3) for k, c in zip(FEATURES, coefs)},
    })
    mask = df.truncated_by_depth & (df.in_deg > 0)
    if report["cv_auc_mean"] < MIN_AUC:
        # честно: признаки не различают «ушли дальше» и «осели» → не делаем вид, что знаем
        report["status"] = f"слабая модель (AUC<{MIN_AUC}) — для всех обрезанных берём базовую долю"
        df.loc[mask, "p_continue"] = report["base_rate_has_out"]
    elif mask.any():
        df.loc[mask, "p_continue"] = model.predict_proba(_X(df[mask]))[:, 1]
    report["n_truncated"] = int(df.truncated_by_depth.sum())
    report["expected_true_terminals"] = float((1 - df.loc[mask, "p_continue"]).sum())
    return df, report
