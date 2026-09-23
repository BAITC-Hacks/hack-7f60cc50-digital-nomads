#!/usr/bin/env python3
"""
Досье на клиента: факты из графа → Markdown (+ аналитическая записка LLM, если задан LLM_API_KEY).

    python investigate.py --gid <gid>                # досье на клиента
    python investigate.py --seeds <seed_gid_1> <seed_gid_2>   # кто собирает деньги этих seed
    python investigate.py --gid ... --no-llm                      # только факты, без LLM

Нужен out/graph.json (его пишет run.py). Результат печатается и сохраняется в out/cases/.
"""
import argparse
import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gm import assistant, investigate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out", help="папка с результатами run.py")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gid", help="gid клиента")
    g.add_argument("--seeds", nargs="+", help="gid seed-клиентов: досье на их общего получателя")
    ap.add_argument("--no-llm", action="store_true", help="не вызывать LLM даже при заданном ключе")
    a = ap.parse_args()
    gj = Path(a.out) / "graph.json"
    if not gj.exists():
        sys.exit(f"Нет {gj} — сначала запустите: python run.py --data data --out {a.out}")
    gi = assistant.GraphIndex(json.loads(gj.read_text(encoding="utf-8")))
    try:
        r = investigate.investigate(gi, gid=a.gid, seeds=a.seeds, use_llm=False if a.no_llm else None)
    except ValueError as e:
        sys.exit(f"Ошибка: {e}")
    d = Path(a.out) / "cases"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"case_{r['target']}.md"
    p.write_text(r["markdown"], encoding="utf-8")
    print(r["markdown"])
    print(f"\n[режим: {r['mode']}; сохранено: {p}]")


if __name__ == "__main__":
    main()
