"""
AI-ассистент аналитика: вопрос на естественном языке → вызовы функций графа → ответ со ссылками на gid.

Два режима:
  * LLM (если задан LLM_API_KEY): модель сама выбирает функции графа (tool calling) и формулирует ответ.
    Поддержаны OpenAI-совместимые API (LLM_PROVIDER=openai, LLM_BASE_URL, LLM_MODEL)
    и Anthropic Messages API (LLM_PROVIDER=anthropic).
  * Детерминированный (без ключа): gid извлекаются из вопроса, намерение — по ключевым словам.

Контроль галлюцинаций: каждый gid в ответе LLM сверяется с тем, что вернули функции графа
(ground_gids). Выдуманный или искажённый gid помечается в ответе и в интерфейсе.
Роли и приоритеты LLM НЕ назначает — только читает посчитанные пайплайном факты.
Зависимости: только стандартная библиотека Python.
"""
import json
import math
import os
import re
import urllib.request
from collections import deque

ROLE_RU = {"coordinator": "координатор", "consolidator": "консолидатор", "distributor": "распределитель",
           "transit": "транзит", "terminal": "конечный получатель", "peripheral": "периферия"}
ROLES = list(ROLE_RU)
PATTERN_RU = {
    "drop_like": "дроп: быстрый сквозной пропуск 1–2 получателям",
    "exchanger_like": "похож на P2P-обменник/хаб",
    "cash_out_candidate": "кандидат на вывод: деньги seed пришли и не ушли внутри банка",
    "external_funding": "источник средств вне выборки (нужен KYC)",
    "retail_collection": "розничный сбор: много мелких поступлений",
}
MAX_LIST = 15          # элементов в списке результата функции (обрезаем по элементам, не по символам JSON)
GID_RE = re.compile(r"(?<!\d)\d{15,20}(?!\d)")

SYSTEM = """Ты помощник AML-аналитика банка (Казахстан). Данные: граф внутрибанковских переводов ≥5 000 KZT за месяц,
построенный от 81 seed-клиента (связаны с незаконным оборотом наркотиков) на 4 колена по ИСХОДЯЩИМ переводам.

Правила:
1. Опирайся только на результаты функций графа. Каждый gid в ответе копируй дословно из результатов функций —
   18 цифр, не сокращай и не округляй. Не упоминай gid, которых в результатах не было.
2. Формулируй выводы как гипотезы для проверки («признаки консолидации»), никогда не утверждай виновность.
3. Не придумывай данные о клиентах (ФИО, возраст, адреса) — их в выгрузке нет.
4. Учитывай ограничения данных: у seed входящие занижены (граф строился от них); у узлов 4-го колена исходящие
   не выгружались (p_continue — оценка модели); переводы < 5 000 KZT, межбанк и наличные не видны.
5. Роли: coordinator — к узлу по хронологии сходятся деньги нескольких seed и он управляет структурой;
   consolidator — собирает от многих; distributor — веерная рассылка; transit — пропускает ≈100% за короткий срок;
   terminal — деньги пришли и не ушли внутри банка; peripheral — выраженной роли нет.
6. Отвечай по-русски, кратко: 3–8 пунктов, суммы в тенге, в конце — что проверить дальше."""


def kzt(x):
    x = float(x or 0)
    return f"{x / 1e6:.1f} млн" if x >= 1e6 else (f"{x / 1e3:.0f} тыс." if x >= 1e3 else f"{x:.0f}")


def _num(x, default=-1.0):
    try:
        v = float(x)
        return default if math.isnan(v) else v
    except (TypeError, ValueError):
        return default


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
        self.TX = {}
        for a, b, d, s in payload.get("tx", []):
            self.TX.setdefault((int(a), int(b)), []).append((int(d), s))
        self.period_start = payload.get("period_start")
        self._sim = None

    def _need(self, gid) -> int:
        if not re.fullmatch(r"\s*\d{1,20}\s*", str(gid)):
            raise ValueError(f"gid должен состоять из цифр, получено {gid!r}")
        g = int(str(gid).strip())
        if g not in self.N:
            raise ValueError(f"gid {gid} не найден в графе")
        return g

    def _gids(self, gids):
        if isinstance(gids, (str, int)):
            gids = re.findall(r"\d+", str(gids))
        return [int(g) for g in gids if int(g) in self.N]

    def _day(self, d):
        return f"день {d + 1}" if not self.period_start else f"{self.period_start[:8]}{d + 1:02d}"

    # ------------------------------------------------ функции графа (они же tools для LLM)
    def node_card(self, gid) -> dict:
        g = self._need(gid)
        n = self.N[g]
        top_in = sorted(self.IN.get(g, []), key=lambda x: -x[1])[:8]
        top_out = sorted(self.OUT.get(g, []), key=lambda x: -x[1])[:8]
        prio = {k[5:]: n.get(k) for k in n if k.startswith("prio_")}
        return {
            "gid": g, "role": n["role"], "role_ru": ROLE_RU[n["role"]], "role_score": n["role_score"],
            "priority_score": n["priority_score"], "priority_rank": n.get("priority_rank"),
            "priority_breakdown": prio, "cluster_id": n["cluster_id"], "is_seed": n["is_seed"],
            "depth": n["depth"], "evidence": n["evidence"], "in_kzt": n["in_kzt"], "out_kzt": n["out_kzt"],
            "in_deg": n["in_deg"], "out_deg": n["out_deg"], "in_tx": n.get("in_tx"), "out_tx": n.get("out_tx"),
            "pass_through": n["pass_through"], "seed_reach_chronological": n["seed_reach"],
            "seed_reach_static": n.get("seed_reach_static"), "seed_share_in": n.get("seed_share_in"),
            "merge_gain": n["merge_gain"], "patterns": [PATTERN_RU.get(p, p) for p in
                                                       str(n.get("pattern") or "none").split(";") if p != "none"],
            "truncated_by_depth": n["truncated_by_depth"], "p_continue": n["p_continue"],
            "fast_out_share": n.get("fast_out_share"), "weak_seed_link": n.get("weak_seed_link"),
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
        gids = self._gids(gids)
        max_hops = max(1, min(int(max_hops), 4))
        cnt, hops = {}, {}
        for g in gids:
            for v, d in self._bfs(g, self.OUT, max_hops).items():
                cnt[v] = cnt.get(v, 0) + 1
                hops.setdefault(v, []).append(d)
        rows = [(v, c) for v, c in cnt.items() if c >= 2 or len(gids) == 1]
        rows.sort(key=lambda x: (-x[1], -self.N[x[0]]["priority_score"]))
        return {"sources": gids, "max_hops": max_hops, "receivers": [
            {"gid": v, "reached_from": c, "min_hops": min(hops[v]), "role": self.N[v]["role"],
             "priority_score": self.N[v]["priority_score"], "evidence": self.N[v]["evidence"]}
            for v, c in rows[:MAX_LIST]], "truncated": len(rows) > MAX_LIST}

    def upstream(self, gid, max_hops: int = 3) -> dict:
        """Откуда пришли деньги: плательщики выше по потоку."""
        g = self._need(gid)
        d = self._bfs(g, self.IN, max(1, min(int(max_hops), 4)))
        rows = sorted(d.items(), key=lambda x: (x[1], -self.N[x[0]]["priority_score"]))
        return {"gid": g, "payers": [{"gid": v, "hops": h, "role": self.N[v]["role"],
                                      "is_seed": self.N[v]["is_seed"]} for v, h in rows[:MAX_LIST]],
                "n_upstream": len(d), "n_seed_upstream": sum(1 for v in d if self.N[v]["is_seed"]),
                "truncated": len(rows) > MAX_LIST}

    def _chrono_prev(self, src):
        """Самое раннее прибытие денег из src в каждый узел по хронологии переводов (даты не убывают)."""
        if not hasattr(self, "_tx_sorted"):
            self._tx_sorted = sorted((d, a, b) for (a, b), lst in self.TX.items() for d, _ in lst)
        arr, prev = {src: -1}, {src: None}
        i, T = 0, self._tx_sorted
        while i < len(T):
            j = i
            while j < len(T) and T[j][0] == T[i][0]:
                j += 1
            changed = True
            while changed:          # цепочки внутри одного дня
                changed = False
                for d, a, b in T[i:j]:
                    if a in arr and arr[a] <= d and b not in arr:
                        arr[b], prev[b] = d, a
                        changed = True
            i = j
        return prev

    def money_path(self, src, dst) -> dict:
        """Путь денег src → dst с суммами и датами каждого звена. Сначала ищется путь по хронологии
        переводов (деньги могли пройти именно так); если его нет — кратчайший путь в графе с пометкой."""
        src, dst = self._need(src), self._need(dst)
        prev = self._chrono_prev(src) if self.TX else {}
        if dst not in prev:
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
            days = sorted(d for d, _ in self.TX.get((a, b), []))
            legs.append({"from": a, "to": b, "sum_kzt": s,
                         "dates": [self._day(d) for d in days[:3]] + (["…"] if len(days) > 3 else [])})
        chrono = all(min(d for d, _ in self.TX.get((a, b), [(0, 0)])) <= max(d for d, _ in self.TX.get((b, c), [(99, 0)]))
                     for (a, b), (_, c) in zip(zip(path, path[1:]), zip(path[1:], path[2:])))
        return {"src": src, "dst": dst, "path": path, "legs": legs, "chronological": chrono,
                "roles": {g: self.N[g]["role"] for g in path}}

    def edge_timeline(self, src, dst) -> dict:
        """Все переводы src → dst с датами и суммами."""
        a, b = self._need(src), self._need(dst)
        t = sorted(self.TX.get((a, b), []))
        return {"src": a, "dst": b, "n_tx": len(t), "sum_kzt": sum(s for _, s in t),
                "transfers": [{"date": self._day(d), "sum_kzt": s} for d, s in t[:MAX_LIST]],
                "truncated": len(t) > MAX_LIST}

    def top_nodes(self, role: str = None, n: int = 10) -> dict:
        role = (role or "").strip().lower() or None
        if role and role not in ROLE_RU:
            return {"error": f"роль {role!r} неизвестна; допустимы: {', '.join(ROLES)}"}
        n = max(1, min(int(n), MAX_LIST))
        rows = [x for x in self.P["nodes"] if not role or x["role"] == role]
        rows.sort(key=lambda x: -x["priority_score"])
        return {"role": role, "nodes": [{"gid": x["gid"], "role": x["role"], "priority_score": x["priority_score"],
                                         "evidence": x["evidence"]} for x in rows[:n]]}

    def cluster_info(self, cluster_id) -> dict:
        for c in self.P["clusters"]:
            if int(c["cluster_id"]) == int(cluster_id):
                members = sorted((x for x in self.P["nodes"] if int(x["cluster_id"]) == int(cluster_id)),
                                 key=lambda x: -x["priority_score"])
                return {**c, "top_members": [{"gid": x["gid"], "role": x["role"],
                                              "priority_score": x["priority_score"]} for x in members[:10]]}
        return {"error": "нет такого кластера"}

    def gaps_for(self, gid) -> dict:
        """Каких данных не хватает по клиенту и что запросить дальше (белые пятна)."""
        g = self._need(gid)
        rows = [r for r in self.P.get("gaps", []) if str(r.get("gid")) == str(g)]
        n = self.N[g]
        if n["is_seed"]:
            rows.append({"gap": "seed", "observation": "входящие из-за пределов выборки не видны",
                         "next_request": "Выгрузить все входящие переводы клиента за период"})
        return {"gid": g, "gaps": rows}

    def find_similar(self, gid, k: int = 5) -> dict:
        """Узлы с похожим поведением (стандартизованные метрики, евклидово расстояние)."""
        g = self._need(gid)
        feats = ["in_deg", "out_deg", "in_kzt", "out_kzt", "fast_out_share", "seed_reach", "merge_gain", "in_tx"]
        if self._sim is None:
            cols = []
            for f in feats:
                xs = [math.log1p(max(_num(x.get(f), 0), 0)) for x in self.P["nodes"]]
                m = sum(xs) / len(xs)
                sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5 or 1.0
                cols.append([(x - m) / sd for x in xs])
            self._sim = {x["gid"]: [c[i] for c in cols] for i, x in enumerate(self.P["nodes"])}
        me = self._sim[g]
        dist = sorted((sum((a - b) ** 2 for a, b in zip(me, v)) ** 0.5, h) for h, v in self._sim.items() if h != g)
        k = max(1, min(int(k), 10))
        return {"gid": g, "features": feats, "similar": [
            {"gid": h, "distance": round(d, 2), "role": self.N[h]["role"], "evidence": self.N[h]["evidence"]}
            for d, h in dist[:k]]}


def _schema(props, required):
    return {"type": "object", "properties": props, "required": required}


GID = {"type": "string", "description": "gid клиента — ровно 18 цифр, копировать дословно"}
TOOLS = [
    ("node_card", "Карточка клиента: роль, скор, обоснование, метрики, разложение приоритета, типологии, "
                  "крупнейшие плательщики и получатели.", {"gid": GID}, ["gid"]),
    ("common_receivers", "Кто собирает деньги с нескольких клиентов: общие получатели ниже по потоку.",
     {"gids": {"type": "array", "items": GID}, "max_hops": {"type": "integer", "minimum": 1, "maximum": 4}}, ["gids"]),
    ("upstream", "Откуда пришли деньги к клиенту: плательщики выше по потоку, в т.ч. seed.",
     {"gid": GID, "max_hops": {"type": "integer", "minimum": 1, "maximum": 4}}, ["gid"]),
    ("money_path", "Кратчайший направленный путь денег от src к dst с суммами и датами; "
                   "chronological=false значит, что даты звеньев не складываются в движение одних денег.",
     {"src": GID, "dst": GID}, ["src", "dst"]),
    ("edge_timeline", "Все переводы от src к dst с датами и суммами.", {"src": GID, "dst": GID}, ["src", "dst"]),
    ("top_nodes", "Топ клиентов по приоритету, опционально с фильтром по роли.",
     {"role": {"type": "string", "enum": ROLES}, "n": {"type": "integer", "minimum": 1, "maximum": MAX_LIST}}, []),
    ("cluster_info", "Сводка по кластеру: размер, seed, оборот, гипотеза, главные участники.",
     {"cluster_id": {"type": "integer"}}, ["cluster_id"]),
    ("gaps_for", "Каких данных не хватает по клиенту и что запросить дальше.", {"gid": GID}, ["gid"]),
    ("find_similar", "Клиенты с похожим поведением — «кто ещё ведёт себя так же».",
     {"gid": GID, "k": {"type": "integer", "minimum": 1, "maximum": 10}}, ["gid"]),
]
TOOL_NAMES = {n for n, *_ in TOOLS}


def _call(gi: GraphIndex, name: str, args):
    """Выполнить tool-вызов модели. Ошибки (неизвестная функция, кривые аргументы) возвращаются
    модели как результат — она может исправиться на следующем шаге, а ответ не падает целиком."""
    if name not in TOOL_NAMES:
        return {"error": f"нет функции {name}; доступны: {', '.join(sorted(TOOL_NAMES))}"}
    if not isinstance(args, dict):
        return {"error": f"аргументы {name} должны быть JSON-объектом"}
    try:
        return getattr(gi, name)(**args)
    except (TypeError, ValueError, KeyError) as e:
        return {"error": f"неверные аргументы {name}: {e}"}


def _dump(res) -> str:
    return json.dumps(res, ensure_ascii=False, default=str)


# ------------------------------------------------ контроль галлюцинаций
def _collect_gids(obj, acc):
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_gids(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_gids(v, acc)
    elif isinstance(obj, (int, str)):
        acc.update(int(m) for m in GID_RE.findall(str(obj)))
    return acc


def ground_gids(answer: str, grounded: set, gi: GraphIndex, question: str = "") -> dict:
    """Каждый gid в ответе должен прийти из результатов функций (или из вопроса) и существовать в графе."""
    allowed = set(grounded) | {int(g) for g in GID_RE.findall(question or "")}
    found = [int(g) for g in dict.fromkeys(GID_RE.findall(answer or ""))]
    return {"verified": [g for g in found if g in allowed and g in gi.N],
            "not_from_tools": [g for g in found if g not in allowed and g in gi.N],
            "not_in_graph": [g for g in found if g not in gi.N],
            "ok": all(g in allowed and g in gi.N for g in found)}


# ------------------------------------------------ LLM-режим
def _http(url, headers, body, timeout=90):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={**headers, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def llm_configured():
    return bool(os.environ.get("LLM_API_KEY"))


def _args(raw):
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError as e:
        return f"невалидный JSON аргументов: {e}"


def _llm_openai(gi, question, system=SYSTEM, max_steps=6):
    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    hdr = {"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"}
    tools = [{"type": "function", "function": {"name": n, "description": d, "parameters": _schema(p, r)}}
             for n, d, p, r in TOOLS]
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    used, grounded = [], set()
    for step in range(max_steps + 1):
        last = step == max_steps
        body = {"model": model, "messages": msgs, "temperature": 0, "max_tokens": 2000}
        if not last:
            body["tools"] = tools
        else:  # шаги кончились — просим итог по уже собранным фактам, а не теряем их
            msgs.append({"role": "user", "content": "Лимит вызовов исчерпан. Дай итоговый ответ по собранным фактам."})
        resp = _http(f"{base}/chat/completions", hdr, body)
        m = resp["choices"][0]["message"]
        msgs.append(m)
        if not m.get("tool_calls"):
            return m.get("content") or "", used, grounded
        for tc in m["tool_calls"]:
            name = tc["function"]["name"]
            args = _args(tc["function"].get("arguments"))
            used.append(name)
            res = _call(gi, name, args) if isinstance(args, dict) else {"error": args}
            _collect_gids(res, grounded)
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": _dump(res)})
    return "", used, grounded


def _llm_anthropic(gi, question, system=SYSTEM, max_steps=6):
    model = os.environ.get("LLM_MODEL", "claude-sonnet-5")
    base = os.environ.get("LLM_BASE_URL", "https://api.anthropic.com").rstrip("/")
    hdr = {"x-api-key": os.environ["LLM_API_KEY"], "anthropic-version": "2023-06-01"}
    tools = [{"name": n, "description": d, "input_schema": _schema(p, r)} for n, d, p, r in TOOLS]
    msgs = [{"role": "user", "content": question}]
    used, grounded = [], set()
    for step in range(max_steps + 1):
        last = step == max_steps
        # запас на рассуждения модели: у современных моделей мышление тратит тот же лимит токенов
        body = {"model": model, "max_tokens": 8000, "system": system, "tools": tools, "messages": msgs}
        if last:
            body["tool_choice"] = {"type": "none"}
        resp = _http(f"{base}/v1/messages", hdr, body)
        stop = resp.get("stop_reason")
        if stop == "refusal":
            raise RuntimeError("модель отказалась отвечать (stop_reason=refusal)")
        content = resp.get("content", [])
        msgs.append({"role": "assistant", "content": content})
        calls = [b for b in content if b.get("type") == "tool_use"]
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        if stop == "pause_turn":
            continue
        if not calls:
            if stop == "max_tokens" and not text:
                raise RuntimeError("ответ обрезан лимитом токенов")
            return text, used, grounded
        results = []
        for b in calls:
            used.append(b["name"])
            res = _call(gi, b["name"], b.get("input"))
            _collect_gids(res, grounded)
            results.append({"type": "tool_result", "tool_use_id": b["id"], "content": _dump(res)})
        msgs.append({"role": "user", "content": results})
    return "", used, grounded


def run_llm(gi, question, system=SYSTEM):
    prov = os.environ.get("LLM_PROVIDER", "openai").lower()
    fn = _llm_anthropic if prov == "anthropic" else _llm_openai
    text, used, grounded = fn(gi, question, system)
    if not text.strip():
        raise RuntimeError("модель вернула пустой ответ")
    return text, used, grounded, prov


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
        legs = "\n".join(f"  {l['from']} → {l['to']}: {kzt(l['sum_kzt'])} KZT ({', '.join(l['dates'])})"
                         for l in r["legs"])
        roles = ", ".join(f"{x} ({ROLE_RU[r['roles'][x]]})" for x in r["path"])
        chrono = "" if r["chronological"] else "\nВнимание: даты звеньев не складываются в движение одних и тех же денег."
        return f"Путь денег ({len(r['path']) - 1} шага):\n{legs}\nУчастники: {roles}{chrono}", ["money_path"]
    # «откуда» проверяем раньше получателей: оно содержит подстроку «куда»
    if g and any(w in ql for w in ["откуда", "кто платит", "плательщ", "источник"]):
        r = gi.upstream(g[0], 3)
        lines = [f"• {x['gid']} ({ROLE_RU[x['role']]}{', seed' if x['is_seed'] else ''}), {x['hops']} кол. выше"
                 for x in r["payers"][:10]]
        return (f"Выше по потоку от {g[0]}: {r['n_upstream']} клиентов, из них seed {r['n_seed_upstream']}.\n"
                + "\n".join(lines)), ["upstream"]
    if g and any(w in ql for w in ["похож", "так же", "аналог"]):
        r = gi.find_similar(g[0], 5)
        return ("Похожие по поведению клиенты:\n" + "\n".join(
            f"• {x['gid']} ({ROLE_RU[x['role']]}, расстояние {x['distance']}): {x['evidence']}" for x in r["similar"])
        ), ["find_similar"]
    if g and any(w in ql for w in ["запрос", "не хватает", "белые пятна", "пробел"]):
        r = gi.gaps_for(g[0])
        if not r["gaps"]:
            return f"По {g[0]} отдельных белых пятен не отмечено.", ["gaps_for"]
        return (f"Что запросить по {g[0]}:\n" + "\n".join(
            f"• {x['gap']}: {x['observation']} → {x['next_request']}" for x in r["gaps"])), ["gaps_for"]
    if g and (len(g) >= 2 or any(w in ql for w in ["собира", "кому", "куда", "получател"])):
        r = gi.common_receivers(g, 3)
        if not r["receivers"]:
            return "Общих получателей в пределах 3 колен не найдено.", ["common_receivers"]
        lines = [f"• {x['gid']} — {ROLE_RU[x['role']]}, деньги приходят от {x['reached_from']} из {len(g)} "
                 f"(от {x['min_hops']} кол.), приоритет {x['priority_score']:.3f}: {x['evidence']}"
                 for x in r["receivers"][:6]]
        return ("Признаки консолидации — узлы ниже по потоку, куда сходятся деньги указанных клиентов:\n"
                + "\n".join(lines) + "\nЭто гипотезы для проверки."), ["common_receivers"]
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
    try:
        c = gi.node_card(gid)
    except ValueError as e:
        return str(e)
    pay = ", ".join(f"{x['gid']} ({kzt(x['sum_kzt'])})" for x in c["top_payers"][:5]) or "нет"
    rec = ", ".join(f"{x['gid']} ({kzt(x['sum_kzt'])})" for x in c["top_receivers"][:5]) or "нет"
    notes = []
    if c["truncated_by_depth"]:
        notes.append("узел на границе выгрузки — исходящие не видны, нужен 5-й уровень")
    if c["is_seed"]:
        notes.append("seed: входящие из-за пределов выборки не видны")
    notes += c["patterns"]
    return (f"Клиент {c['gid']}: {c['role_ru']} (уверенность {c['role_score']}), приоритет {c['priority_score']}"
            f"{' (№' + str(c['priority_rank']) + ')' if c.get('priority_rank') else ''}, "
            f"кластер {c['cluster_id']}, колено {c['depth']}.\nОбоснование: {c['evidence']}\n"
            f"Получил {kzt(c['in_kzt'])} от {c['in_deg']}, отправил {kzt(c['out_kzt'])} на {c['out_deg']}.\n"
            f"Крупнейшие плательщики: {pay}\nКрупнейшие получатели: {rec}\n"
            + (f"На что обратить внимание: {'; '.join(notes)}\n" if notes else "")
            + "Выводы — гипотезы для проверки, а не утверждение о причастности.")


def answer(gi: GraphIndex, question: str) -> dict:
    if llm_configured():
        try:
            text, used, grounded, prov = run_llm(gi, question)
            return {"answer": text, "tools": used, "mode": f"LLM ({prov})",
                    "grounding": ground_gids(text, grounded, gi, question)}
        except Exception as e:  # сеть/ключ/лимиты — не роняем интерфейс
            text, used = _rule_based(gi, question)
            return {"answer": text, "tools": used, "mode": f"детерминированный (LLM недоступен: {e})"}
    text, used = _rule_based(gi, question)
    return {"answer": text, "tools": used, "mode": "детерминированный (LLM_API_KEY не задан)"}


_CARD_CACHE = {}


def card(gi: GraphIndex, gid: int) -> dict:
    base = card_text(gi, gid)
    if not llm_configured():
        return {"answer": base, "mode": "шаблон (LLM_API_KEY не задан)"}
    if gid not in _CARD_CACHE:   # каждый клик по узлу не должен заново гонять LLM-цикл
        q = (f"Составь краткую справку по клиенту {gid} для AML-аналитика (5–7 предложений): роль, потоки, "
             f"связи, на что обратить внимание. Используй node_card и при необходимости upstream.")
        _CARD_CACHE[gid] = answer(gi, q)
    r = _CARD_CACHE[gid]
    return {"answer": r["answer"], "mode": r["mode"], "grounding": r.get("grounding")}
