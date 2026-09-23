"""
Тест HTTP-сервера serve.py: нормальные запросы, мусорный ввод и защита от чужих страниц.

    python tests/test_serve.py        # или: pytest -q
"""
import http.client
import json
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import serve  # noqa: E402
from gm import assistant  # noqa: E402
from test_assistant import PAYLOAD, S1, S2  # noqa: E402


def _start():
    d = tempfile.mkdtemp()
    (Path(d) / "viewer.html").write_text("<html>ok</html>", encoding="utf-8")
    gi = assistant.GraphIndex(json.loads(json.dumps(PAYLOAD)))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), None)
    srv.RequestHandlerClass = serve.make_handler(Path(d), gi, srv.server_port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _req(port, method, path, body=None, headers=None, host=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    h = {"Host": host or f"localhost:{port}", **(headers or {})}
    c.request(method, path, body=body, headers=h)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def test_serve():
    srv = _start()
    p = srv.server_port
    J = {"Content-Type": "application/json"}
    try:
        assert _req(p, "GET", "/")[0] == 200
        st, b = _req(p, "GET", "/api/health")
        assert st == 200 and json.loads(b)["ok"] is True
        st, b = _req(p, "GET", f"/api/card?gid={S1}")
        assert st == 200 and str(S1) in json.loads(b)["answer"]
        q = json.dumps({"question": f"Кто собирает деньги с {S1}, {S2}?"}).encode()
        st, b = _req(p, "POST", "/api/ask", q, {**J, "Origin": f"http://localhost:{p}"})
        assert st == 200 and json.loads(b)["tools"] == ["common_receivers"]
        # мусорный ввод → понятная ошибка, а не оборванное соединение
        for gid in ["abc", "1" * 50, "%C2%B2", "-5"]:
            assert _req(p, "GET", f"/api/card?gid={gid}")[0] == 400, gid
        assert _req(p, "POST", "/api/ask", b'{"question": 123}', J)[0] == 400
        assert _req(p, "POST", "/api/ask", "\xca\xf2".encode("latin-1"), J)[0] == 400
        assert _req(p, "POST", "/api/ask", b"x" * 70000, J)[0] == 413
        assert _req(p, "GET", "/nope")[0] == 404
        # чужие страницы: DNS rebinding, кросс-доменный POST, «простой» text/plain без preflight
        assert _req(p, "GET", "/", host="attacker.example")[0] == 403
        assert _req(p, "POST", "/api/ask", q, {**J, "Origin": "https://evil.example"})[0] == 403
        assert _req(p, "POST", "/api/ask", q, {"Content-Type": "text/plain"})[0] == 415
    finally:
        srv.shutdown()


if __name__ == "__main__":
    test_serve()
    print("OK")
