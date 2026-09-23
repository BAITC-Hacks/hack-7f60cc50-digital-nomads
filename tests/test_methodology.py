"""
Юнит-тесты ключевой логики на крошечных графах, где правильный ответ считается в уме.

    python tests/test_methodology.py        # или: pytest -q
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gm import boundary, export, features, io  # noqa: E402
from gm.config import Thresholds  # noqa: E402

S1, S2, A, B, C, D = 1, 2, 10, 11, 12, 13


def _tx(rows):
    return pd.DataFrame(rows, columns=["src", "dst", "date", "sum_kzt"]).assign(
        date=lambda t: pd.to_datetime(t.date))


def test_temporal_reach_respects_chronology():
    # S1→A 20-го, A→B 5-го: статический путь есть, но это не движение одних денег
    # S1→A 20-го, A→C 20-го: тот же день допустим
    tx = _tx([(S1, A, "2026-07-20", 1e5), (A, B, "2026-07-05", 1e5), (A, C, "2026-07-20", 1e5)])
    reach, carried = features.temporal_reach(tx, {S1})
    assert reach[A] == {S1} and reach[C] == {S1}
    assert B not in reach
    assert carried[(A, C)] == {S1} and (A, B) not in carried


def test_merge_gain_counts_branches_by_date():
    # D получает от A (несёт S1) и от B (несёт S2) → сводит 2 потока, сверх любой ветви +1
    # C получает от A раньше, чем деньги S1 дошли до A → ветвь A→C seed не несёт
    tx = _tx([(S1, A, "2026-07-02", 1e5), (S2, B, "2026-07-02", 1e5),
              (A, D, "2026-07-03", 1e5), (B, D, "2026-07-03", 1e5), (A, C, "2026-07-01", 1e5)])
    edges = tx.groupby(["src", "dst"]).sum_kzt.agg(["sum", "size"]).reset_index()
    edges.columns = ["src", "dst", "sum_kzt", "n_tx"]
    edges["depth"] = 1
    nodes = pd.DataFrame({"gid": [S1, S2, A, B, C, D], "depth": [0, 0, 1, 1, 2, 2],
                          "is_seed": [True, True, False, False, False, False]})
    G = features.build_graph(edges, nodes)
    df = features.provenance(G, features.structural(G, nodes), tx).set_index("gid")
    assert df.loc[D, "seed_reach"] == 2 and df.loc[D, "merge_gain"] == 1
    assert df.loc[C, "seed_reach"] == 0 and df.loc[C, "seed_reach_static"] == 1


def test_boundary_features_ignore_backward_edges():
    # у узла 2-го колена есть входящий от узла 3-го (обратный поток) — у 4-го колена таких нет,
    # поэтому модель не должна его видеть
    df = pd.DataFrame({"gid": [S1, A, B], "depth": [0, 2, 3], "out_deg": [1, 1, 1], "out_kzt": [1e5, 1e5, 1e5]})
    tx = _tx([(S1, A, "2026-07-01", 1e5), (B, A, "2026-07-02", 9e5)])
    f = boundary.forward_features(df, tx).set_index(df.gid)
    assert f.loc[A, "in_deg_fwd"] == 1
    assert np.isclose(f.loc[A, "log_in_kzt_fwd"], np.log1p(1e5))


def test_boundary_zero_base_rate_is_not_replaced():
    # у всех узлов 1–3 колена нет исходящих → базовая доля 0.0, а не «0 or 0.5» = 0.5
    n = 10
    df = pd.DataFrame({"gid": range(100, 100 + n), "depth": [1] * (n - 1) + [4], "is_seed": False,
                       "out_deg": 0, "out_kzt": 0.0, "truncated_by_depth": [False] * (n - 1) + [True]})
    tx = _tx([(1, g, "2026-07-01", 1e4) for g in df.gid])
    df.loc[len(df)] = [1, 0, True, n, 1e5, False]
    df, rep = boundary.fit_boundary_model(df, tx)
    assert rep["base_rate_has_out"] == 0.0
    assert df.loc[df.truncated_by_depth, "p_continue"].tolist() == [0.0]


def test_thresholds_load_checks_types():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "th.json"
        p.write_text(json.dumps({"top_n": "40", "cons_min_payers": 6}), encoding="utf-8")
        t = Thresholds.load(p)
        assert t.top_n == 40 and isinstance(t.top_n, int)
        p.write_text(json.dumps({"top_n": "много"}), encoding="utf-8")
        try:
            Thresholds.load(p)
            raise AssertionError("ожидалась ошибка типа")
        except TypeError as e:
            assert "top_n" in str(e)
        p.write_text(json.dumps({"no_such": 1}), encoding="utf-8")
        try:
            Thresholds.load(p)
            raise AssertionError("ожидалась ошибка имени")
        except KeyError:
            pass


def test_float_gid_is_rejected():
    df = pd.DataFrame({"gid": [100000008165763100.0]})
    try:
        io._as_gid(df, "nodes", "gid")
        raise AssertionError("float gid должен отклоняться")
    except ValueError as e:
        assert "float" in str(e)


def test_validate_rejects_empty_cells():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        pd.DataFrame({"gid": [1], "role": ["peripheral"], "role_score": [0.5], "cluster_id": [0],
                      "priority_score": [0.1], "evidence": ["Вход 0 от 0"], "extra": [np.nan]}
                     ).to_csv(out / "nodes_roles.csv", index=False)
        pd.DataFrame({"cluster_id": [0], "n_nodes": [1], "n_seed": [0], "sum_kzt_internal": [0.0],
                      "top_gids": ["1"], "hypothesis": ["вне сети"]}).to_csv(out / "clusters.csv", index=False)
        pd.DataFrame({"rank": [1], "gid": [1], "role": ["peripheral"], "priority_score": [0.1],
                      "why": ["x"]}).to_csv(out / "top_nodes.csv", index=False)
        try:
            export.validate(out, 1, {1})
            raise AssertionError("пустая ячейка должна ронять проверку")
        except AssertionError as e:
            assert "extra" in str(e)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
    print("OK")
