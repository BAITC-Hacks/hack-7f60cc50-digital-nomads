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
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Windows-консоль (cp1251/cp866) не умеет печатать «≈», «→» и т.п. — принудительно UTF-8
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gm import assistant, investigate  # noqa: E402


MAX_BODY = 64 * 1024
MAX_QUESTION = 2000
GID_RE = re.compile(r"[0-9]{1,20}")


def make_handler(out: Path, gi, port: int):
    viewer = (out / "viewer.html").read_bytes()
    # Сервер слушает только 127.0.0.1, но браузер может прийти на него с чужой страницы:
    #  * DNS rebinding — чужое имя, резолвящееся в 127.0.0.1 → проверяем заголовок Host;
    #  * «простой» кросс-доменный POST (text/plain без preflight) тратит LLM-ключ → требуем
    #    application/json (он вызывает CORS preflight, который мы не разрешаем) и свой Origin.
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    allowed_origins = {f"http://{h}" for h in allowed_hosts}

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _host_ok(self):
            if self.headers.get("Host", "") not in allowed_hosts:
                self._send(403, {"error": "недопустимый Host; откройте http://localhost:%d" % port})
                return False
            return True

        def do_GET(self):
            if not self._host_ok():
                return
            try:
                u = urlparse(self.path)
                if u.path in ("/", "/index.html", "/viewer.html"):
                    return self._send(200, viewer, "text/html; charset=utf-8")
                if u.path == "/api/card":
                    gid = parse_qs(u.query).get("gid", [""])[0]
                    if not GID_RE.fullmatch(gid):
                        return self._send(400, {"error": "gid должен быть числом (до 20 цифр)"})
                    return self._send(200, assistant.card(gi, int(gid)))
                if u.path == "/api/health":
                    return self._send(200, {"ok": True, "llm": assistant.llm_configured()})
                self._send(404, {"error": "not found"})
            except Exception as e:  # не рвём соединение молча
                self._send(500, {"error": f"внутренняя ошибка: {type(e).__name__}: {e}"})

        def do_POST(self):
            if not self._host_ok():
                return
            route = urlparse(self.path).path
            if route not in ("/api/ask", "/api/investigate"):
                return self._send(404, {"error": "not found"})
            origin = self.headers.get("Origin")
            if origin is not None and origin not in allowed_origins:
                return self._send(403, {"error": "запрос с чужой страницы отклонён"})
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._send(415, {"error": "ожидается Content-Type: application/json"})
            try:
                n = int(self.headers.get("Content-Length", 0))
            except ValueError:
                n = -1
            if not 0 <= n <= MAX_BODY:
                return self._send(413, {"error": f"тело запроса должно быть от 0 до {MAX_BODY} байт"})
            try:
                body = json.loads((self.rfile.read(n) or b"{}").decode("utf-8"))
                if not isinstance(body, dict):
                    raise AttributeError
            except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                return self._send(400, {"error": 'ожидается JSON-объект в UTF-8'})
            if route == "/api/investigate":
                gid, seeds = body.get("gid"), body.get("seeds")
                if gid is not None and not GID_RE.fullmatch(str(gid)):
                    return self._send(400, {"error": "gid должен быть числом (до 20 цифр)"})
                if gid is None and not (isinstance(seeds, list) and seeds and len(seeds) <= 50
                                        and all(GID_RE.fullmatch(str(x)) for x in seeds)):
                    return self._send(400, {"error": "нужен gid или seeds — список от 1 до 50 gid"})
                try:
                    return self._send(200, investigate.investigate(gi, gid=gid, seeds=seeds))
                except ValueError as e:
                    return self._send(400, {"error": str(e)})
                except Exception as e:
                    return self._send(500, {"error": f"внутренняя ошибка: {type(e).__name__}: {e}"})
            q = body.get("question", "")
            if not isinstance(q, str) or not q.strip() or len(q) > MAX_QUESTION:
                return self._send(400, {"error": f"question — непустая строка до {MAX_QUESTION} символов"})
            try:
                self._send(200, assistant.answer(gi, q))
            except Exception as e:
                self._send(500, {"error": f"внутренняя ошибка: {type(e).__name__}: {e}"})

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
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(out, gi, a.port)).serve_forever()


if __name__ == "__main__":
    main()
