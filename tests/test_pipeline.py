"""
Смоук-тест пайплайна на синтетических данных (реальные данные не нужны).
    python tests/test_pipeline.py        # или: pytest -q
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Windows-консоль (cp1251/cp866) не умеет печатать «≈», «→» и т.п. — принудительно UTF-8
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gm import boundary  # noqa: E402
from gm.config import Thresholds  # noqa: E402


def test_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        data, out = Path(d) / "data", Path(d) / "out"
        subprocess.run([sys.executable, ROOT / "tools/make_synthetic.py", "--out", data, "--csv"], check=True)
        subprocess.run([sys.executable, ROOT / "run.py", "--data", data, "--out", out], check=True)
        nodes = pd.read_csv(data / "nodes.csv")
        nr = pd.read_csv(out / "nodes_roles.csv")
        cl = pd.read_csv(out / "clusters.csv")
        tp = pd.read_csv(out / "top_nodes.csv")
        assert len(nr) == len(nodes) and set(nr.gid) == set(nodes.gid)
        assert nr[["role", "role_score", "cluster_id", "priority_score", "evidence"]].notna().all().all()
        assert set(nr.cluster_id) <= set(cl.cluster_id)
        assert len(tp) >= 20 and tp.priority_score.is_monotonic_decreasing
        # ловушка №1: узел 4-го колена без исходящих не может быть «наблюдаемым» стоком
        t4 = nr[nr.truncated_by_depth & (nr.role == "terminal")]
        assert (t4.terminal_kind == "estimated").all()
        # транзит не присваивается seed (их входящие занижены)
        assert not ((nr.role == "transit") & nr.is_seed).any()
        assert (out / "viewer.html").exists() and (out / "graph.json").exists()
        # заложенные в синтетику консолидаторы должны находиться
        gt = pd.read_csv(data / "ground_truth.csv").merge(nr[["gid", "role"]], on="gid")
        planted = gt[gt.planted_role == "consolidator"]
        assert planted.role.isin(["consolidator", "coordinator", "transit"]).mean() >= 0.8

        # модель обрыва: новые поля отчёта и документированный порог качества
        b = json.loads((out / "run_report.json").read_text(encoding="utf-8"))["boundary_model"]
        assert "calibration" in b and "feature_importance_ci" in b and "threshold_curve" in b
        assert set(b["feature_importance_ci"]) == set(boundary.FEATURES)
        assert b["terminal_threshold"]["configured"] == Thresholds().terminal_max_p_continue
        _check_quality_gate(b, nr)
        assert (out / "calibration.svg").exists() and "reliability diagram" in (out / "report.md").read_text(encoding="utf-8")


def _check_quality_gate(b, nr):
    """Документированное поведение (README, раздел 9): при CV AUC < MIN_AUC модель не прогнозирует —
    всем обрезанным ставится базовая доля продолжения, и terminal(estimated) появляется, только если
    сама базовая доля ниже порога. Синтетика не содержит сигнала для этой задачи (AUC ≈ 0.53),
    поэтому на ней обязана сработать именно деградация; на реальных данных AUC ≈ 0.66 ≥ 0.6."""
    trunc = nr[nr.truncated_by_depth]
    if b["cv_auc_mean"] < boundary.MIN_AUC:
        assert b["degraded_to_baseline"] and "слабая модель" in b["status"]
        assert np.allclose(trunc.p_continue, b["base_rate_has_out"]), "при деградации p_continue = базовая доля"
        if b["base_rate_has_out"] >= Thresholds().terminal_max_p_continue:
            assert not (nr.terminal_kind == "estimated").any(), "нет сигнала → нет оценённых стоков"
    else:
        assert not b["degraded_to_baseline"]
        assert trunc.p_continue.nunique() > 1


def _toy(signal: bool, n=900, seed=0):
    """Узлы 1-го колена получают от k seed; есть ли исходящие — зависит от k (signal) или случайно.
    Плюс узлы 4-го колена без исходящих (их и оценивает модель)."""
    rng = np.random.default_rng(seed)
    seeds = list(range(1, 11))
    rows, tx, gid = [], [], 1000
    for s in seeds:
        rows.append((s, 0, True, 1, 1e5, False))
    for i in range(n):
        gid += 1
        k = int(rng.integers(1, 7))
        cont = (rng.random() < (0.15 if k <= 3 else 0.85)) if signal else (rng.random() < 0.5)
        for s in rng.choice(seeds, size=k, replace=False):
            tx.append((int(s), gid, "2026-07-02", float(rng.integers(10, 200)) * 1e3))
        rows.append((gid, 1, False, int(cont), 5e4 * cont, False))
    for i in range(60):
        gid += 1
        payer = 1001 + int(rng.integers(0, n))
        tx.append((payer, gid, "2026-07-03", 5e4))
        rows.append((gid, 4, False, 0, 0.0, True))
    df = pd.DataFrame(rows, columns=["gid", "depth", "is_seed", "out_deg", "out_kzt", "truncated_by_depth"])
    t = pd.DataFrame(tx, columns=["src", "dst", "date", "sum_kzt"])
    return df, t.assign(date=pd.to_datetime(t.date))


def test_boundary_learns_when_there_is_signal():
    df, tx = _toy(signal=True)
    df, b = boundary.fit_boundary_model(df, tx)
    assert b["cv_auc_mean"] >= 0.8 and not b["degraded_to_baseline"]
    assert b["calibration"]["ece"] < 0.1                            # 0.3 ≈ 30%, а не «какое-то число»
    assert b["feature_importance_ci"]["in_deg_fwd"]["significant"]  # сигнал найден в нужном признаке
    assert df.loc[df.truncated_by_depth, "p_continue"].nunique() > 1


def test_boundary_degrades_to_baseline_on_noise():
    df, tx = _toy(signal=False)
    df, b = boundary.fit_boundary_model(df, tx)
    assert b["cv_auc_mean"] < boundary.MIN_AUC and b["degraded_to_baseline"]
    p = df.loc[df.truncated_by_depth, "p_continue"]
    assert np.allclose(p, b["base_rate_has_out"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
    print("OK")
