#!/usr/bin/env python3
"""
Локальный сервер экрана просмотра + AI-ассистента. Только стандартная библиотека.

    python serve.py --out out            # затем открыть http://localhost:8765

Переменные окружения для LLM (необязательно; без них — детерминированный режим):
    LLM_API_KEY    ключ
    LLM_PROVIDER   openai (по умолчанию, любой OpenAI-совместимый API) | anthropic
    LLM_BASE_URL   напр. https://api.openai.com/v1
    LLM_MODEL      имя модели
"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gm import assistant  # noqa: E402


def make_handler(out: Path, gi):
    viewer = (out / "viewer.html").read_bytes()

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html", "/viewer.html"):
                return self._send(200, viewer, "text/html; charset=utf-8")
            if u.path == "/api/card":
                gid = parse_qs(u.query).get("gid", ["0"])[0]
                return self._send(200, assistant.card(gi, int(gid)))
            if u.path == "/api/health":
                return self._send(200, {"ok": True, "llm": assistant.llm_configured()})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if urlparse(self.path).path != "/api/ask":
                return self._send(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length", 0))
            q = json.loads(self.rfile.read(n) or b"{}").get("question", "")
            self._send(200, assistant.answer(gi, q))

        def log_message(self, fmt, *args):
            sys.stderr.write("[serve] " + fmt % args + "\n")

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out", help="папка с результатами run.py")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    out = Path(a.out)
    if not (out / "graph.json").exists():
        sys.exit(f"Нет {out}/graph.json — сначала запустите: python run.py --data data --out {out}")
    gi = assistant.GraphIndex(json.loads((out / "graph.json").read_text(encoding="utf-8")))
    print(f"LLM: {'включён' if assistant.llm_configured() else 'не настроен → детерминированный режим'}")
    print(f"Откройте http://localhost:{a.port}")
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(out, gi)).serve_forever()


if __name__ == "__main__":
    main()
