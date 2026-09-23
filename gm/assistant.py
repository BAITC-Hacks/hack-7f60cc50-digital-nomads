"""
AI-ассистент аналитика: вопрос на естественном языке → вызовы функций графа → ответ со ссылками на gid.

Два режима:
  * LLM (если задан LLM_API_KEY): модель сама выбирает функции графа (tool calling) и формулирует ответ.
    Поддержаны OpenAI-совместимые API (LLM_PROVIDER=openai, LLM_BASE_URL, LLM_MODEL)
    и Anthropic Messages API (LLM_PROVIDER=anthropic).
  * Детерминированный (без ключа): gid извлекаются из вопроса, намерение — по ключевым словам.

Роли и приоритеты LLM НЕ назначает — только читает посчитанные пайплайном факты.
Зависимости: только стандартная библиотека Python.
"""
import json
import os
import re
import urllib.request
from collections import deque

ROLE_RU = {"coordinator": "координатор", "consolidator": "консолидатор", "distributor": "распределитель",
           "transit": "транзит", "terminal": "конечный получатель", "peripheral": "периферия"}

SYSTEM = (
    "Ты помощник AML-аналитика банка. Отвечай по-русски, кратко, только по фактам из функций графа. "
    "Всегда указывай gid. Формулируй выводы как гипотезы для проверки («признаки консолидации»), "
    "никогда не утверждай виновность. Не придумывай данные о клиентах (ФИО, возраст и т.п.) — их нет."
)


def kzt(x):
    x = float(x or 0)
    return f"{x / 1e6:.1f} млн" if x >= 1e6 else (f"{x / 1e3:.0f} тыс." if x >= 1e3 else f"{x:.0f}")


class GraphIndex:
    def __init__(self, payload: dict):
        self.P = payload
        # в graph.json gid хранятся строками (точность JS) — внутри ассистента работаем с int
        self.N = {int(n["gid"]): {**n, "gid": int(n["gid"])} for n in payload["nodes"]}
        self.P["nodes"] = list(self.N.values())
        self.OUT, self.IN = {}, {}
        for u, v, s, c in payload["edges"]:
            u, v = int(u), int(v)
            self.OUT.setdefault(u, []).append((v, s, c))
            self.IN.setdefault(v, []).append((u, s, c))

    # ------------------------------------------------ функции графа (они же tools для LLM)
    def node_card(self, gid: int) -> dict:
        n = self.N.get(int(gid))
        if not n:
            return {"error": f"gid {gid} не найден"}
        top_in = sorted(self.IN.get(n["gid"], []), key=lambda x: -x[1])[:8]
        top_out = sorted(self.OUT.get(n["gid"], []), key=lambda x: -x[1])[:8]
        return {
            "gid": n["gid"], "role": n["role"], "role_ru": ROLE_RU[n["role"]], "role_score": n["role_score"],
            "priority_score": n["priority_score"], "cluster_id": n["cluster_id"], "is_seed": n["is_seed"],
            "depth": n["depth"], "evidence": n["evidence"], "in_kzt": n["in_kzt"], "out_kzt": n["out_kzt"],
            "in_deg": n["in_deg"], "out_deg": n["out_deg"], "pass_through": n["pass_through"],
            "seed_reach": n["seed_reach"], "merge_gain": n["merge_gain"],
            "truncated_by_depth": n["truncated_by_depth"], "p_continue": n["p_continue"],
            "top_payers": [{"gid": u, "role": self.N[u]["role"], "sum_kzt": s, "n_tx": c} for u, s, c in top_in],
            "top_receivers": [{"gid": v, "role": self.N[v]["role"], "sum_kzt": s, "n_tx": c} for v, s, c in top_out],
        }

    def _bfs(self, start, adj, max_hops):
        dist = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            if dist[u] >= max_hops:
                continue
            for v, *_ in adj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        dist.pop(start)
        return dist

    def common_receivers(self, gids, max_hops: int = 3) -> dict:
        """Кто собирает деньги с этих клиентов: узлы ниже по потоку, достижимые из нескольких gid."""
        gids = [int(g) for g in gids if int(g) in self.N]
        cnt, hops = {}, {}
        for g in gids:
            for v, d in self._bfs(g, self.OUT, max_hops).items():
                cnt[v] = cnt.get(v, 0) + 1
                hops.setdefault(v, []).append(d)
        rows = [(v, c) for v, c in cnt.items() if c >= 2 or len(gids) == 1]
        rows.sort(key=lambda x: (-x[1], -self.N[x[0]]["priority_score"]))
        return {"sources": gids, "max_hops": max_hops, "receivers": [
            {"gid": v, "reached_from": c, "min_hops": min(hops[v]), "role": self.N[v]["role"],
             "priority_score": self.N[v]["priority_score"], "evidence": self.N[v]["evidence"]} for v, c in rows[:10]]}

    def upstream(self, gid: int, max_hops: int = 3) -> dict:
        """Откуда пришли деньги: плательщики выше по потоку."""
        d = self._bfs(int(gid), self.IN, max_hops)
        rows = sorted(d.items(), key=lambda x: (x[1], -self.N[x[0]]["priority_score"]))[:15]
        return {"gid": int(gid), "payers": [{"gid": v, "hops": h, "role": self.N[v]["role"],
                                             "is_seed": self.N[v]["is_seed"]} for v, h in rows],
                "n_upstream": len(d), "n_seed_upstream": sum(1 for v in d if self.N[v]["is_seed"])}

    def money_path(self, src: int, dst: int) -> dict:
        """Кратчайший путь денег src → dst по направлению переводов."""
        src, dst = int(src), int(dst)
        prev = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                break
            for v, *_ in self.OUT.get(u, []):
                if v not in prev:
                    prev[v] = u
                    q.append(v)
        if dst not in prev:
            return {"src": src, "dst": dst, "path": None, "note": "направленного пути нет"}
        path, x = [], dst
        while x is not None:
            path.append(x)
            x = prev[x]
        path.reverse()
        legs = []
        for a, b in zip(path, path[1:]):
            s = next(s for v, s, _ in self.OUT[a] if v == b)
            legs.append({"from": a, "to": b, "sum_kzt": s})
        return {"src": src, "dst": dst, "path": path, "legs": legs,
                "roles": {g: self.N[g]["role"] for g in path}}

    def top_nodes(self, role: str = None, n: int = 10) -> dict:
        rows = [x for x in self.P["nodes"] if not role or x["role"] == role]
        rows.sort(key=lambda x: -x["priority_score"])
        return {"role": role, "nodes": [{"gid": x["gid"], "role": x["role"], "priority_score": x["priority_score"],
                                         "evidence": x["evidence"]} for x in rows[:int(n)]]}

    def cluster_info(self, cluster_id: int) -> dict:
        for c in self.P["clusters"]:
            if int(c["cluster_id"]) == int(cluster_id):
                return c
        return {"error": "нет такого кластера"}


TOOLS = [
    ("node_card", "Карточка клиента: роль, скор, обоснование, метрики, крупнейшие плательщики и получатели.",
     {"gid": {"type": "string", "description": "gid клиента (число из 18 цифр)"}}, ["gid"]),
    ("common_receivers", "Кто собирает деньги с нескольких клиентов: общие получатели ниже по потоку.",
     {"gids": {"type": "array", "items": {"type": "string"}}, "max_hops": {"type": "integer"}}, ["gids"]),
    ("upstream", "Откуда пришли деньги к клиенту: плательщики выше по потоку, в т.ч. seed.",
     {"gid": {"type": "string"}, "max_hops": {"type": "integer"}}, ["gid"]),
    ("money_path", "Кратчайший направленный путь денег от src к dst с суммами.",
     {"src": {"type": "string"}, "dst": {"type": "string"}}, ["src", "dst"]),
    ("top_nodes", "Топ клиентов по приоритету, опционально с фильтром по роли "
                  "(coordinator, consolidator, distributor, transit, terminal, peripheral).",
     {"role": {"type": "string"}, "n": {"type": "integer"}}, []),
    ("cluster_info", "Сводка по кластеру: размер, seed, оборот, гипотеза.",
     {"cluster_id": {"type": "integer"}}, ["cluster_id"]),
]


def _call(gi: GraphIndex, name: str, args: dict):
    return getattr(gi, name)(**args)


# ------------------------------------------------ LLM-режим
def _http(url, headers, body, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={**headers, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def llm_configured():
    return bool(os.environ.get("LLM_API_KEY"))


def _llm_openai(gi, question, max_steps=5):
    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    hdr = {"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"}
    tools = [{"type": "function", "function": {"name": n, "description": d,
              "parameters": {"type": "object", "properties": p, "required": r}}} for n, d, p, r in TOOLS]
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    used = []
    for _ in range(max_steps):
        resp = _http(f"{base}/chat/completions", hdr, {"model": model, "messages": msgs, "tools": tools, "temperature": 0})
        m = resp["choices"][0]["message"]
        msgs.append(m)
        if not m.get("tool_calls"):
            return m.get("content") or "", used
        for tc in m["tool_calls"]:
            name = tc["function"]["name"]
            args = json.loads(tc["function"].get("arguments") or "{}")
            used.append(name)
            res = _call(gi, name, args)
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(res, ensure_ascii=False)[:12000]})
    return "Превышено число шагов.", used


def _llm_anthropic(gi, question, max_steps=5):
    model = os.environ.get("LLM_MODEL", "claude-sonnet-4-5")
    base = os.environ.get("LLM_BASE_URL", "https://api.anthropic.com").rstrip("/")
    hdr = {"x-api-key": os.environ["LLM_API_KEY"], "anthropic-version": "2023-06-01"}
    tools = [{"name": n, "description": d, "input_schema": {"type": "object", "properties": p, "required": r}}
             for n, d, p, r in TOOLS]
    msgs = [{"role": "user", "content": question}]
    used = []
    for _ in range(max_steps):
        resp = _http(f"{base}/v1/messages", hdr, {"model": model, "max_tokens": 1200, "system": SYSTEM,
                                                  "tools": tools, "messages": msgs})
        msgs.append({"role": "assistant", "content": resp["content"]})
        calls = [b for b in resp["content"] if b["type"] == "tool_use"]
        if not calls:
            return "".join(b.get("text", "") for b in resp["content"] if b["type"] == "text"), used
        results = []
        for b in calls:
            used.append(b["name"])
            res = _call(gi, b["name"], b["input"])
            results.append({"type": "tool_result", "tool_use_id": b["id"],
                            "content": json.dumps(res, ensure_ascii=False)[:12000]})
        msgs.append({"role": "user", "content": results})
    return "Превышено число шагов.", used


# ------------------------------------------------ детерминированный режим
def _gids_in(gi, text):
    return [int(x) for x in re.findall(r"\d{4,}", text) if int(x) in gi.N]


def _rule_based(gi, q):
    ql = q.lower()
    g = _gids_in(gi, q)
    if len(g) >= 2 and any(w in ql for w in ["путь", "как деньги", "от ", "маршрут"]) and not any(
            w in ql for w in ["собира", "общ"]):
        r = gi.money_path(g[0], g[1])
        if not r["path"]:
            return f"Направленного денежного пути от {g[0]} к {g[1]} в выгрузке нет.", ["money_path"]
        legs = "\n".join(f"  {l['from']} → {l['to']}: {kzt(l['sum_kzt'])} KZT" for l in r["legs"])
        roles = ", ".join(f"{x} ({ROLE_RU[r['roles'][x]]})" for x in r["path"])
        return f"Путь денег ({len(r['path']) - 1} шага):\n{legs}\nУчастники: {roles}", ["money_path"]
    if g and (len(g) >= 2 or any(w in ql for w in ["собира", "кому", "куда", "получател"])):
        r = gi.common_receivers(g, 3)
        if not r["receivers"]:
            return "Общих получателей в пределах 3 колен не найдено.", ["common_receivers"]
        lines = [f"• {x['gid']} — {ROLE_RU[x['role']]}, деньги приходят от {x['reached_from']} из {len(g)} "
                 f"(мин. {x['min_hops']} кол.), приоритет {x['priority_score']:.3f}: {x['evidence']}"
                 for x in r["receivers"][:6]]
        return ("Признаки консолидации — узлы ниже по потоку, куда сходятся деньги указанных клиентов:\n"
                + "\n".join(lines) + "\nЭто гипотезы для проверки."), ["common_receivers"]
    if g and any(w in ql for w in ["откуда", "кто платит", "плательщ", "источник"]):
        r = gi.upstream(g[0], 3)
        lines = [f"• {x['gid']} ({ROLE_RU[x['role']]}{', seed' if x['is_seed'] else ''}), {x['hops']} кол. выше"
                 for x in r["payers"][:10]]
        return (f"Выше по потоку от {g[0]}: {r['n_upstream']} клиентов, из них seed {r['n_seed_upstream']}.\n"
                + "\n".join(lines)), ["upstream"]
    if g:
        return card_text(gi, g[0]), ["node_card"]
    for role in ROLE_RU:
        if role in ql or ROLE_RU[role][:6] in ql:
            r = gi.top_nodes(role, 10)
            return "\n".join(f"• {x['gid']} — {x['priority_score']:.3f}: {x['evidence']}" for x in r["nodes"]), ["top_nodes"]
    r = gi.top_nodes(None, 10)
    return ("Не распознал gid в вопросе. Топ-10 по приоритету:\n"
            + "\n".join(f"• {x['gid']} ({ROLE_RU[x['role']]}) — {x['priority_score']:.3f}" for x in r["nodes"])), ["top_nodes"]


def card_text(gi, gid):
    c = gi.node_card(gid)
    if "error" in c:
        return c["error"]
    pay = ", ".join(f"{x['gid']} ({kzt(x['sum_kzt'])})" for x in c["top_payers"][:5]) or "нет"
    rec = ", ".join(f"{x['gid']} ({kzt(x['sum_kzt'])})" for x in c["top_receivers"][:5]) or "нет"
    notes = []
    if c["truncated_by_depth"]:
        notes.append("узел на границе выгрузки — исходящие не видны, нужен 5-й уровень")
    if c["is_seed"]:
        notes.append("seed: входящие из-за пределов выборки не видны")
    return (f"Клиент {c['gid']}: {c['role_ru']} (уверенность {c['role_score']}), приоритет {c['priority_score']}, "
            f"кластер {c['cluster_id']}, колено {c['depth']}.\nОбоснование: {c['evidence']}\n"
            f"Получил {kzt(c['in_kzt'])} от {c['in_deg']}, отправил {kzt(c['out_kzt'])} на {c['out_deg']}.\n"
            f"Крупнейшие плательщики: {pay}\nКрупнейшие получатели: {rec}\n"
            + (f"На что обратить внимание: {'; '.join(notes)}\n" if notes else "")
            + "Выводы — гипотезы для проверки, а не утверждение о причастности.")


def answer(gi: GraphIndex, question: str) -> dict:
    if llm_configured():
        try:
            prov = os.environ.get("LLM_PROVIDER", "openai").lower()
            fn = _llm_anthropic if prov == "anthropic" else _llm_openai
            text, used = fn(gi, question)
            return {"answer": text, "tools": used, "mode": f"LLM ({prov})"}
        except Exception as e:  # сеть/ключ/лимиты — не роняем интерфейс
            text, used = _rule_based(gi, question)
            return {"answer": text, "tools": used, "mode": f"детерминированный (LLM недоступен: {e})"}
    text, used = _rule_based(gi, question)
    return {"answer": text, "tools": used, "mode": "детерминированный (LLM_API_KEY не задан)"}


def card(gi: GraphIndex, gid: int) -> dict:
    base = card_text(gi, gid)
    if not llm_configured():
        return {"answer": base, "mode": "шаблон (LLM_API_KEY не задан)"}
    q = (f"Составь краткую справку по клиенту {gid} для AML-аналитика (5–7 предложений): роль, потоки, "
         f"связи, на что обратить внимание. Используй node_card и при необходимости upstream.")
    r = answer(gi, q)
    return {"answer": r["answer"], "mode": r["mode"]}
