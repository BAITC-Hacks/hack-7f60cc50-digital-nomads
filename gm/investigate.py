"""
Агент расследования: по gid (или списку seed) собирает досье и черновик сообщения о подозрительной
операции для внутреннего кейса.

Устройство — гибрид, в котором LLM не может испортить факты:
  1. collect()  — детерминированный сбор фактов функциями графа. У каждого факта свой id (F1, F2, …)
                  и список gid, на которые он опирается.
  2. render()   — досье в Markdown строится из фактов по шаблону: участники, движение денег с датами,
                  признаки, ограничения данных, что запросить. Работает без LLM.
  3. narrate()  — если задан LLM_API_KEY, модель пишет аналитическую записку, ссылаясь на [F#], и может
                  вызывать функции графа (tool calling) для уточнений.
  4. verify()   — каждый gid записки должен прийти из фактов или результатов функций и существовать
                  в графе; каждая ссылка [F#] — существовать. При ошибках — один раунд исправления
                  со списком ошибок; если не помогло, записка помечается «не подтверждена».

Досье — гипотеза для проверки аналитиком, не вывод о виновности.
"""
import re
from datetime import date

from . import assistant as A

FACT_RE = re.compile(r"\[(F\d+)\]")
TYPOLOGY = {
    "coordinator": "концентрация потоков нескольких seed в одном узле, который управляет структурой (возможный верхний уровень)",
    "consolidator": "сбор средств от множества плательщиков на один счёт (счёт-накопитель)",
    "distributor": "веерная рассылка средств множеству получателей (выплаты дропам/курьерам)",
    "transit": "транзитный счёт: средства уходят дальше почти целиком за короткий срок",
    "terminal": "средства оседают: внутрибанковских исходящих нет (возможен вывод наличными или за периметр)",
}


class Case:
    def __init__(self, gi, target):
        self.gi, self.target = gi, target
        self.facts = []

    def add(self, text, gids=(), data=None):
        fid = f"F{len(self.facts) + 1}"
        self.facts.append({"id": fid, "text": text, "gids": [int(g) for g in gids], "data": data})
        return fid

    @property
    def grounded(self):
        return {g for f in self.facts for g in f["gids"]}


def pick_target(gi, gid=None, seeds=None):
    if gid is not None:
        return gi._need(gid), None
    seeds = gi._gids(seeds or [])
    if len(seeds) < 1:
        raise ValueError("нужен gid или список seed")
    cr = gi.common_receivers(seeds, 3)
    if not cr["receivers"]:
        raise ValueError("у указанных клиентов нет общих получателей в пределах 3 колен")
    return int(cr["receivers"][0]["gid"]), cr


def collect(gi, gid=None, seeds=None, max_paths=3) -> Case:
    target, cr = pick_target(gi, gid, seeds)
    case = Case(gi, target)
    c = gi.node_card(target)
    case.card = c
    if cr:
        case.f_sources = case.add(
            f"Деньги клиентов {', '.join(map(str, cr['sources']))} сходятся ниже по потоку; узел {target} "
            f"достижим из {cr['receivers'][0]['reached_from']} из них.", cr["sources"] + [target], cr)
    rank = f" (№{c['priority_rank']} из {len(gi.N)})" if c.get("priority_rank") else ""
    case.f_role = case.add(
        f"Клиент {target}: роль «{c['role_ru']}» (уверенность {c['role_score']}), приоритет {c['priority_score']}{rank}, "
        f"колено {c['depth']}, кластер {c['cluster_id']}. Обоснование пайплайна: {c['evidence']}", [target], c)
    case.f_flow = case.add(
        f"Получено {A.kzt(c['in_kzt'])} KZT от {c['in_deg']} плательщиков ({c['in_tx']} переводов), "
        f"отправлено {A.kzt(c['out_kzt'])} KZT {c['out_deg']} получателям ({c['out_tx']} переводов).", [target])

    # пути от seed по хронологии
    seeds_up = [int(s) for s in str(gi.N[target].get("reach_seeds") or "").split(";") if s.strip().isdigit()]
    case.f_paths = []
    for s in seeds_up[:max_paths]:
        p = gi.money_path(s, target)
        if not p["path"]:
            continue
        legs = "; ".join(f"{l['from']}→{l['to']} {A.kzt(l['sum_kzt'])} ({', '.join(l['dates'])})" for l in p["legs"])
        note = "" if p["chronological"] else " Даты звеньев кратчайшего пути не упорядочены — это не прямое движение одних денег."
        case.f_paths.append(case.add(f"Путь от seed {s}: {legs}.{note}", p["path"], p))
    case.n_seeds_up = len(seeds_up)

    # крупнейшие контрагенты с датами
    case.f_cp = []
    for x in c["top_payers"][:3]:
        t = gi.edge_timeline(x["gid"], target)
        dates = ", ".join(tr["date"] for tr in t["transfers"][:4]) + ("…" if t["n_tx"] > 4 else "")
        case.f_cp.append(case.add(
            f"Плательщик {x['gid']} ({A.ROLE_RU[x['role']]}): {t['n_tx']} перев. на {A.kzt(t['sum_kzt'])} KZT ({dates}).",
            [x["gid"], target], t))
    for x in c["top_receivers"][:3]:
        t = gi.edge_timeline(target, x["gid"])
        dates = ", ".join(tr["date"] for tr in t["transfers"][:4]) + ("…" if t["n_tx"] > 4 else "")
        case.f_cp.append(case.add(
            f"Получатель {x['gid']} ({A.ROLE_RU[x['role']]}): {t['n_tx']} перев. на {A.kzt(t['sum_kzt'])} KZT ({dates}).",
            [target, x["gid"]], t))

    cl = gi.cluster_info(c["cluster_id"])
    case.f_cluster = None
    if "error" not in cl and int(c["cluster_id"]) != 0:
        case.f_cluster = case.add(
            f"Кластер {c['cluster_id']}: {cl.get('n_nodes')} клиентов, {cl.get('n_seed')} seed; "
            f"гипотеза: {cl.get('hypothesis')}", [target] + [m["gid"] for m in cl["top_members"][:3]], cl)
    sim = gi.find_similar(target, 3)
    case.f_similar = case.add(
        "Похожее поведение у: " + ", ".join(f"{x['gid']} ({A.ROLE_RU[x['role']]})" for x in sim["similar"]) + ".",
        [x["gid"] for x in sim["similar"]], sim)
    gaps = gi.gaps_for(target)["gaps"]
    case.f_gaps = [case.add(f"{g['gap']}: {g['observation']} → {g['next_request']}", [target], g) for g in gaps]
    return case


def render(case: Case, narrative=None) -> str:
    gi, c, t = case.gi, case.card, case.target
    f = {x["id"]: x for x in case.facts}
    L = [f"# Досье: клиент {t}", "",
         f"*Сформировано {date.today():%d.%m.%Y} автоматически по выгрузке переводов "
         f"(период с {gi.period_start or '—'}). Черновик для внутреннего кейса — проверяется аналитиком. "
         f"Гипотеза для проверки, не вывод о виновности.*", "",
         "## 1. Суть подозрения", ""]
    if getattr(case, "f_sources", None):
        L.append(f"- {f[case.f_sources]['text']} [{case.f_sources}]")
    L += [f"- {f[case.f_role]['text']} [{case.f_role}]",
          f"- {f[case.f_flow]['text']} [{case.f_flow}]",
          f"- По хронологии переводов к клиенту доходят деньги {case.n_seeds_up} seed-клиентов."]
    if c["role"] in TYPOLOGY:
        L.append(f"- Типология по роли: {TYPOLOGY[c['role']]}.")
    for p in c["patterns"]:
        L.append(f"- Дополнительный признак: {p}.")
    if narrative:
        L += ["", "## 2. Аналитическая записка (LLM)", "",
              ("✅ Все gid и ссылки на факты проверены." if narrative["verification"]["ok"] else
               "⚠️ Записка не прошла проверку: " + narrative["verification"]["summary"] + ". Опирайтесь на разделы 1, 3–6."),
              "", narrative["text"].strip()]
    n = 3 if narrative else 2
    L += ["", f"## {n}. Участники", "", "| gid | роль | приоритет | колено | связь |", "|---|---|---|---|---|"]
    seen = []
    for fid in [case.f_role] + case.f_paths + case.f_cp:
        for g in f[fid]["gids"]:
            if g not in seen:
                seen.append(g)
    for g in seen[:15]:
        m = gi.N[g]
        link = "объект досье" if g == t else ("seed" if m["is_seed"] else "контрагент")
        L.append(f"| {g} | {A.ROLE_RU[m['role']]} | {m['priority_score']:.3f} | {m['depth']} | {link} |")
    L += ["", f"## {n + 1}. Движение денег", ""]
    L += [f"- {f[x]['text']} [{x}]" for x in case.f_paths] or ["- Хронологически допустимых путей от seed не найдено."]
    L += [f"- {f[x]['text']} [{x}]" for x in case.f_cp]
    if case.f_cluster:
        L.append(f"- {f[case.f_cluster]['text']} [{case.f_cluster}]")
    L.append(f"- {f[case.f_similar]['text']} [{case.f_similar}]")
    L += ["", f"## {n + 2}. Ограничения данных", "",
          "- Видны только внутрибанковские переводы ≥ 5 000 KZT; наличные, межбанк, карты и мелкие суммы не видны.",
          "- Граф построен от seed по исходящим: входящие seed-клиентов из-за пределов выборки отсутствуют.",
          "- У клиентов 4-го колена исходящие не выгружались; вероятность продолжения — оценка модели."
          + (f" Для этого клиента P(ушли дальше) = {c['p_continue']:.2f}." if c["truncated_by_depth"] and
             c["p_continue"] not in (None, -1) else ""),
          "- Сведений о клиентах (KYC) в выгрузке нет: легальные объяснения не исключены."]
    L += ["", f"## {n + 3}. Что запросить", ""]
    L += [f"- {f[x]['text']} [{x}]" for x in case.f_gaps] or ["- Отдельных белых пятен не отмечено."]
    L += ["- KYC: вид деятельности, заявленный доход, дата открытия счёта, устройства и IP входа в банкинг.",
          "", f"## {n + 4}. Для сообщения о подозрительной операции", "",
          "Поля ниже заполняются аналитиком по форме, действующей в банке; досье даёт факты, а не решение.", "",
          f"- **Субъект:** клиент {t} (идентификационные данные — из клиентской базы).",
          f"- **Период операций:** месяц выгрузки, с {gi.period_start or '—'}.",
          f"- **Объём:** получено {A.kzt(c['in_kzt'])} KZT, отправлено {A.kzt(c['out_kzt'])} KZT.",
          f"- **Признаки:** {c['role_ru']}; " + ("; ".join(c["patterns"]) or "дополнительных типологий нет") + ".",
          "- **Основание:** связь денежными потоками с клиентами из списка правоохранительных органов (seed).",
          "- **Важно:** не информировать клиента о проверке.", "",
          "## Факты", ""]
    L += [f"- **{x['id']}** — {x['text']}" for x in case.facts]
    return "\n".join(L) + "\n"


NARRATE_SYSTEM = A.SYSTEM + """

Сейчас ты пишешь аналитическую записку к досье. Тебе даны пронумерованные факты [F1], [F2], …
Каждое утверждение записки подкрепляй ссылкой на факт в квадратных скобках, например [F3].
gid бери только из фактов или из результатов функций графа. 5–8 предложений: гипотеза о роли клиента
в схеме, ключевые связи, что делает её сильнее или слабее, что проверить в первую очередь."""


def verify(text, case: Case, extra_grounded=()) -> dict:
    g = A.ground_gids(text, case.grounded | set(extra_grounded), case.gi)
    ids = {x["id"] for x in case.facts}
    refs = FACT_RE.findall(text)
    bad_refs = sorted(set(r for r in refs if r not in ids))
    problems = []
    if g["not_in_graph"]:
        problems.append("gid нет в графе: " + ", ".join(map(str, g["not_in_graph"])))
    if g["not_from_tools"]:
        problems.append("gid не из фактов: " + ", ".join(map(str, g["not_from_tools"])))
    if bad_refs:
        problems.append("несуществующие ссылки: " + ", ".join(bad_refs))
    if not refs:
        problems.append("нет ни одной ссылки на факты")
    return {"ok": not problems, "summary": "; ".join(problems) or "ok", "gids": g, "fact_refs": sorted(set(refs))}


def narrate(case: Case):
    facts = "\n".join(f"[{x['id']}] {x['text']}" for x in case.facts)
    q = f"Досье на клиента {case.target}. Факты:\n{facts}\n\nНапиши аналитическую записку."
    text, used, grounded, prov = A.run_llm(case.gi, q, NARRATE_SYSTEM)
    v = verify(text, case, grounded)
    rounds = 1
    if not v["ok"]:   # один раунд исправления со списком конкретных ошибок
        q2 = q + f"\n\nПредыдущий вариант записки:\n{text}\n\nИсправь ошибки: {v['summary']}. Не добавляй новых gid."
        text2, used2, grounded2, _ = A.run_llm(case.gi, q2, NARRATE_SYSTEM)
        v2 = verify(text2, case, grounded | grounded2)
        rounds, used = 2, used + used2
        if v2["ok"] or len(v2["summary"]) < len(v["summary"]):
            text, v = text2, v2
    return {"text": text, "verification": v, "tools": used, "mode": f"LLM ({prov})", "rounds": rounds}


def investigate(gi, gid=None, seeds=None, use_llm=None) -> dict:
    case = collect(gi, gid, seeds)
    use_llm = A.llm_configured() if use_llm is None else use_llm
    narrative, mode, err = None, "детерминированный (без LLM)", None
    if use_llm:
        try:
            narrative = narrate(case)
            mode = narrative["mode"]
        except Exception as e:  # сеть/ключ/лимиты — досье всё равно строится из фактов
            err = str(e)
            mode = f"детерминированный (LLM недоступен: {e})"
    return {"target": case.target, "mode": mode, "markdown": render(case, narrative),
            "facts": [{k: v for k, v in x.items() if k != "data"} for x in case.facts],
            "verification": narrative["verification"] if narrative else None,
            "tools": narrative["tools"] if narrative else [], "error": err}


def write_cases(out, payload: dict, n: int = 5, use_llm=False):
    """Досье на топ-n узлов в out/cases/. По умолчанию без LLM — воспроизводимо и бесплатно."""
    from pathlib import Path
    gi = A.GraphIndex(payload)
    d = Path(out) / "cases"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("case_*.md"):
        old.unlink()
    written = []
    for row in payload["top"][:max(0, n)]:
        r = investigate(gi, gid=row["gid"], use_llm=use_llm)
        p = d / f"case_{r['target']}.md"
        p.write_text(r["markdown"], encoding="utf-8")
        written.append(p)
    return written
