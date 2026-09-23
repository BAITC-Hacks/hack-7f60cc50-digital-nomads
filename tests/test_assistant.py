"""
Тест AI-ассистента без реального API-ключа.

Поднимается локальный фейковый LLM-сервер, который говорит на протоколах Anthropic Messages и
OpenAI Chat Completions: на первом шаге «модель» вызывает функции графа (включая заведомо ошибочный
вызов), на втором — формулирует ответ из результатов. Так проверяется весь агентный цикл
assistant.py: сериализация tools, исполнение вызовов, возврат tool_result, обработка ошибок.

    python tests/test_assistant.py        # или: pytest -q
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gm import assistant  # noqa: E402

# Мини-граф: два seed платят консолидатору C, тот — координатору K.
S1, S2, C, K = 100000000000000001, 100000000000000002, 100000000000000003, 100000000000000004


def _node(gid, role, seed, prio):
    return {"gid": str(gid), "role": role, "role_score": 0.7, "priority_score": prio, "cluster_id": 1,
            "is_seed": seed, "depth": 0 if seed else 1, "evidence": f"тест {role}", "in_kzt": 1e5, "out_kzt": 1e5,
            "in_deg": 1, "out_deg": 1, "pass_through": 1.0, "seed_reach": 2, "merge_gain": 1,
            "truncated_by_depth": False, "p_continue": None}


PAYLOAD = {
    "nodes": [_node(S1, "peripheral", True, 0.2), _node(S2, "peripheral", True, 0.2),
              _node(C, "consolidator", False, 0.6), _node(K, "coordinator", False, 0.8)],
    "edges": [[str(S1), str(C), 50000, 2], [str(S2), str(C), 70000, 3], [str(C), str(K), 110000, 1]],
    "clusters": [{"cluster_id": 1, "n_nodes": 4, "hypothesis": "тест"}],
}


class FakeLLM(BaseHTTPRequestHandler):
    """Шаг 1: вызвать common_receivers + несуществующую функцию. Шаг 2: ответ по tool_result."""
    seen = []
    hallucinate = False

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeLLM.seen.append((self.path, body))
        anthropic = self.path.endswith("/v1/messages")
        msgs = body["messages"]
        if anthropic:
            results = [b for m in msgs if m["role"] == "user" and isinstance(m["content"], list)
                       for b in m["content"] if b.get("type") == "tool_result"]
        else:
            results = [m for m in msgs if m["role"] == "tool"]
        if not results:
            calls = [("common_receivers", {"gids": [str(S1), str(S2)]}), ("no_such_tool", {})]
            if anthropic:
                out = {"content": [{"type": "text", "text": "смотрю граф"}] + [
                    {"type": "tool_use", "id": f"t{i}", "name": n, "input": a} for i, (n, a) in enumerate(calls)],
                    "stop_reason": "tool_use"}
            else:
                out = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                    {"id": f"t{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
                    for i, (n, a) in enumerate(calls)]}}]}
        else:
            got = json.loads(results[0]["content"])
            top = got["receivers"][0]["gid"]
            err = json.loads(results[1]["content"])["error"]
            text = f"Деньги обоих seed сходятся на {top}. (ошибка инструмента обработана: {err[:20]})"
            if FakeLLM.hallucinate:   # «модель» выдумала gid и исказила существующий в одной цифре
                text += f" Также связан 100000000000000099 и {str(S1)[:-1]}9."
            out = ({"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"} if anthropic
                   else {"choices": [{"message": {"role": "assistant", "content": text}}]})
        data = json.dumps(out, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def _run_provider(provider):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}" + ("" if provider == "anthropic" else "/v1")
    env = {"LLM_API_KEY": "test", "LLM_PROVIDER": provider, "LLM_BASE_URL": base, "LLM_MODEL": "fake"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    FakeLLM.seen.clear()
    try:
        r = assistant.answer(assistant.GraphIndex(json.loads(json.dumps(PAYLOAD))), f"Кто собирает деньги с {S1}, {S2}?")
    finally:
        srv.shutdown()
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return r


def test_llm_anthropic_tool_loop():
    r = _run_provider("anthropic")
    assert r["mode"] == "LLM (anthropic)", r
    assert r["tools"] == ["common_receivers", "no_such_tool"]
    assert r["grounding"]["ok"], r["grounding"]
    # консолидатор C — единственный общий получатель на 1 шаг; K тоже достижим, но C ближе и сортируется выше по охвату
    assert str(C) in r["answer"] or str(K) in r["answer"]
    assert "ошибка инструмента обработана" in r["answer"]
    first = FakeLLM.seen[0][1]
    assert {t["name"] for t in first["tools"]} == assistant.TOOL_NAMES and "input_schema" in first["tools"][0]


def test_grounding_catches_invented_gids():
    FakeLLM.hallucinate = True
    try:
        r = _run_provider("anthropic")
    finally:
        FakeLLM.hallucinate = False
    g = r["grounding"]
    assert not g["ok"]
    assert 100000000000000099 in g["not_in_graph"]
    assert int(str(S1)[:-1] + "9") in g["not_in_graph"]
    assert g["verified"] and all(x in (C, K) for x in g["verified"])


def test_llm_openai_tool_loop():
    r = _run_provider("openai")
    assert r["mode"] == "LLM (openai)", r
    assert r["tools"] == ["common_receivers", "no_such_tool"]
    assert "ошибка инструмента обработана" in r["answer"]
    assert FakeLLM.seen[0][0] == "/v1/chat/completions"


def test_llm_unreachable_falls_back():
    os.environ.update({"LLM_API_KEY": "x", "LLM_PROVIDER": "anthropic", "LLM_BASE_URL": "http://127.0.0.1:9"})
    try:
        r = assistant.answer(assistant.GraphIndex(json.loads(json.dumps(PAYLOAD))), f"Кто собирает деньги с {S1}, {S2}?")
    finally:
        for k in ("LLM_API_KEY", "LLM_PROVIDER", "LLM_BASE_URL"):
            os.environ.pop(k, None)
    assert r["mode"].startswith("детерминированный (LLM недоступен")
    assert str(C) in r["answer"]


def test_rule_based_modes():
    gi = assistant.GraphIndex(json.loads(json.dumps(PAYLOAD)))
    t, used = assistant._rule_based(gi, f"Кто собирает деньги с {S1}, {S2}?")
    assert used == ["common_receivers"] and str(C) in t and str(K) in t
    t, used = assistant._rule_based(gi, f"Путь от {S1} к {K}")
    assert used == ["money_path"] and "2 шага" in t
    t, used = assistant._rule_based(gi, f"Откуда деньги у {K}?")
    assert used == ["upstream"] and "seed 2" in t
    assert assistant._call(gi, "node_card", {"gid": "abc"})["error"].startswith("неверные аргументы")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
    print("OK")
