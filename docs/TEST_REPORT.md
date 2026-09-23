# Отчёт о проверке

Дата проверки: **23 сентября 2026**. Среда: Windows, Python 3.12.14 из `.audit-venv`.

## Автотесты

Каждый тестовый файл запускался отдельно его штатной командой:

```powershell
& .\.audit-venv\Scripts\python.exe tests/test_methodology.py
& .\.audit-venv\Scripts\python.exe tests/test_pipeline.py
& .\.audit-venv\Scripts\python.exe tests/test_assistant.py
& .\.audit-venv\Scripts\python.exe tests/test_serve.py
& .\.audit-venv\Scripts\python.exe tests/test_investigate.py
```

Реальный консольный итог:

```text
RUN tests/test_methodology.py
ok  test_temporal_reach_respects_chronology
ok  test_merge_gain_counts_branches_by_date
ok  test_boundary_features_ignore_backward_edges
ok  test_boundary_zero_base_rate_is_not_replaced
ok  test_thresholds_load_checks_types
ok  test_float_gid_is_rejected
ok  test_validate_rejects_empty_cells
OK
RESULT tests/test_methodology.py exit=0 elapsed=2.733s

RUN tests/test_pipeline.py
ok  test_end_to_end
ok  test_boundary_learns_when_there_is_signal
ok  test_boundary_degrades_to_baseline_on_noise
OK
RESULT tests/test_pipeline.py exit=0 elapsed=26.491s

RUN tests/test_assistant.py
ok  test_llm_anthropic_tool_loop
ok  test_grounding_catches_invented_gids
ok  test_llm_openai_tool_loop
ok  test_llm_unreachable_falls_back
ok  test_rule_based_modes
OK
RESULT tests/test_assistant.py exit=0 elapsed=3.872s

RUN tests/test_serve.py
OK
RESULT tests/test_serve.py exit=0 elapsed=0.776s

RUN tests/test_investigate.py
ok  test_dossier_without_llm
ok  test_seeds_mode_picks_common_receiver
ok  test_llm_narrative_verified
ok  test_llm_narrative_repaired_or_flagged
OK
RESULT tests/test_investigate.py exit=0 elapsed=1.812s
```

`test_serve.py` дополнительно подтвердил ожидаемые HTTP-коды для `/`, `/api/health`,
`/api/card`, `/api/ask`, ошибочных gid, неверного content type, слишком большого тела,
неизвестного маршрута и запрещённого Origin/Host.

## Объяснение ролей

Команды:

```powershell
& .\.audit-venv\Scripts\python.exe explain.py --gid 100000007908818100 --out out
& .\.audit-venv\Scripts\python.exe explain.py --gid 100000001616816100 --out out
& .\.audit-venv\Scripts\python.exe explain.py --gid 100000004891562100 --out out
```

Все завершились с `exit=0`: `coordinator`, `consolidator` и `terminal (estimated)` соответственно.
Для estimated terminal команда явно проверила `truncated_by_depth=true`,
`p_continue=0.336 < 0.35` и порог входящей суммы.

## Повторный запуск пайплайна

Точная команда:

```powershell
& .\.audit-venv\Scripts\python.exe run.py --data data --out test_agent_out
```

Результат: `exit=0`; внутреннее время пайплайна **9.62 с**, wall time **12.201 с**.
Схема выгрузок прошла встроенную проверку:
`nodes_roles=2248`, `clusters=45`, `top_nodes=30`.

Сравнение SHA256 с эталонной папкой `out`:

| Файл | SHA256 | Совпадает |
|---|---|---:|
| `nodes_roles.csv` | `cc0af5a4014ec005cb691418511ebb3a3c66c1b17ee8fd914fdb213d8f25ada9` | да |
| `clusters.csv` | `a7cb9eb44533f2596e62c071254c91a2f6b8329b8866bde3542f0867f603a104` | да |
| `top_nodes.csv` | `3d46421bc5f1e9eee6c90dbb6dc6a69112f5217e46ba82e9edfb076c729a04dc` | да |

## Viewer, поиск и досье

Сервер запущен командой:

```powershell
& .\.audit-venv\Scripts\python.exe serve.py --out test_agent_out --port 8765
```

Детерминированная выборка `random.Random(20260923).sample(gids, 3)` и результат API:

| gid | `/api/card` | Роль / результат |
|---:|---:|---|
| `100000002878467100` | HTTP 200 | terminal, priority 0.1457, rank 1295, cluster 2 |
| `100000001526235100` | HTTP 200 | peripheral, priority 0.3987, rank 266, cluster 24 |
| `100000008715887100` | HTTP 200 | peripheral, priority 0.0270, rank 2134, cluster 9 |

`/api/investigate` также вернул HTTP 200 и непустое Markdown-досье в детерминированном режиме.

Живая browser-проверка root-agent: **3/3 gid найдены подряд**, для каждого включился
`mode=ego`, на странице присутствовали три canvas-слоя (`canvas=3`). Вкладка «Досье»
сформировала **3436 символов**. Ошибки JavaScript в консоли: **`[]`**.

Итог: все автоматические и живые проверки зелёные.
