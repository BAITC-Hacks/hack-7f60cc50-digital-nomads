"""
Смоук-тест пайплайна на синтетических данных (реальные данные не нужны).
    python tests/test_pipeline.py        # или: pytest -q
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Windows-консоль (cp1251/cp866) не умеет печатать «≈», «→» и т.п. — принудительно UTF-8
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]


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


if __name__ == "__main__":
    test_end_to_end()
    print("OK")
