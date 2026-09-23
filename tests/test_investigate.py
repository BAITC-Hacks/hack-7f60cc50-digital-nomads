"""
Агент расследования: досье из фактов, записка LLM со ссылками [F#] и её проверка.
Реальный API не нужен — фейковый LLM-сервер отвечает заранее заданным текстом.

    python tests/test_investigate.py        # или: pytest -q
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from gm import assistant, investigate  # noqa: E402
from test_assistant import PAYLOAD, S1, S2, C, K  # noqa: E402


def _gi():
    p = json.loads(json.dumps(PAYLOAD))
    for n in p["nodes"]:
        n.update(in_tx=2, out_tx=1, reach_seeds=f"{S1};{S2}" if int(n["gid"]) in (C, K) else "")
    p["tx"] = [[str(S1), str(C), 1, 50000], [str(S2), str(C), 2, 70000], [str(C), str(K), 3, 110000]]
    p["period_start"] = "2026-07-01"
    return assistant.GraphIndex(p)


class Narrator(BaseHTTPRequestHandler):
    replies = []          # очередь текстов записки: по одному на запрос
    calls = 0

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        Narrator.calls += 1
        text = Narrator.replies.pop(0) if Narrator.replies else "пусто"
        data = json.dumps({"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def _with_llm(replies, fn):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Narrator)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    Narrator.replies, Narrator.calls = list(replies), 0
    env = {"LLM_API_KEY": "t", "LLM_PROVIDER": "anthropic", "LLM_BASE_URL": f"http://127.0.0.1:{srv.server_port}"}
    os.environ.update(env)
    try:
        return fn()
    finally:
        srv.shutdown()
        for k in env:
            os.environ.pop(k, None)


def test_dossier_without_llm():
    r = investigate.investigate(_gi(), gid=K, use_llm=False)
    md = r["markdown"]
    assert r["target"] == K and "Досье: клиент" in md
    assert f"Путь от seed {S1}" in md and "2026-07-02" in md          # путь с датами
    assert "не информировать клиента" in md and "Факты" in md
    assert all(f["id"] in md for f in r["facts"])


def test_seeds_mode_picks_common_receiver():
    r = investigate.investigate(_gi(), seeds=[S1, S2], use_llm=False)
    assert r["target"] == K          # оба общие получатели, K выше по приоритету
    assert "Деньги клиентов" in r["markdown"]


def test_llm_narrative_verified():
    ok = f"Клиент {K} получает деньги от {C} [F4], к которому сходятся оба seed [F3]."
    r = _with_llm([ok], lambda: investigate.investigate(_gi(), gid=K))
    assert r["verification"]["ok"], r["verification"]
    assert "✅" in r["markdown"] and ok in r["markdown"]
    assert Narrator.calls == 1


def test_llm_narrative_repaired_or_flagged():
    bad = f"Клиент {K} связан с 100000000000000077 [F99]."
    fixed = f"Клиент {K} получает деньги от {C} [F4]."
    r = _with_llm([bad, fixed], lambda: investigate.investigate(_gi(), gid=K))
    assert r["verification"]["ok"] and Narrator.calls == 2            # исправлено со второго раунда
    r = _with_llm([bad, bad], lambda: investigate.investigate(_gi(), gid=K))
    v = r["verification"]
    assert not v["ok"] and "F99" in v["summary"] and "100000000000000077" in v["summary"]
    assert "⚠️" in r["markdown"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
    print("OK")
