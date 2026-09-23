"""
Все пороги в одном месте. Каждый порог — одна строка README (раздел «Критерии ролей»).
Переопределить без правки кода:  python run.py --config my_thresholds.json
"""
from dataclasses import dataclass, asdict, fields
import json


@dataclass
class Thresholds:
    # --- consolidator: аккумулирует средства от нескольких участников
    cons_min_payers: int = 5            # ≥ N разных плательщиков
    cons_min_in_pct: float = 0.75       # и входящий оборот не ниже P75 среди получателей

    # --- distributor: веерная рассылка
    dist_min_receivers: int = 10        # ≥ N разных получателей
    dist_min_fan_ratio: float = 3.0     # получателей / плательщиков ≥ R

    # --- transit: пропускает, не удерживая
    transit_pass_lo: float = 0.8        # out/in в диапазоне [lo, hi]
    transit_pass_hi: float = 1.2
    transit_max_degree: int = 4         # не более N контрагентов с каждой стороны (иначе это хаб)
    transit_min_in_kzt: float = 50_000  # отсекаем «копеечный» транзит
    fast_days: int = 2                  # «сквозной» перевод: ушло в течение N дней после поступления

    # --- terminal: деньги пришли и остались
    terminal_min_in_kzt: float = 100_000  # или ≥2 плательщиков — иначе это разовый мелкий получатель
    terminal_max_p_continue: float = 0.35 # для 4-го колена: P(у узла есть невидимые исходящие) ниже порога

    # --- coordinator: кандидат в организаторы (второй проход, после базовых ролей)
    coord_min_seed_reach: int = 5       # к узлу сходятся деньги ≥ N разных seed
    coord_min_in_pct: float = 0.97      # входящий оборот в топ-3%
    coord_min_struct_payers: int = 2    # ≥ N плательщиков сами имеют «структурные» роли

    # --- аномалии / признаки
    structuring_lo: float = 5_000       # «дробление у порога»: суммы в [lo, hi)
    structuring_hi: float = 10_000
    structuring_min_share: float = 0.5  # доля таких входящих ≥ share при ≥ 4 переводах
    anomaly_z: float = 3.5              # робастный z-score внутри колена

    # --- кластеризация
    louvain_resolution: float = 1.0
    louvain_seed: int = 42

    # --- топ-лист
    top_n: int = 30
    seed_priority_factor: float = 0.85  # seed уже известны — слегка приглушаем, фокус на новых узлах

    @classmethod
    def load(cls, path=None):
        t = cls()
        if path:
            with open(path, encoding="utf-8") as f:
                over = json.load(f)
            names = {f.name for f in fields(cls)}
            for k, v in over.items():
                if k not in names:
                    raise KeyError(f"неизвестный порог: {k}")
                setattr(t, k, v)
        return t

    def as_dict(self):
        return asdict(self)


ROLES = ["coordinator", "consolidator", "distributor", "transit", "terminal", "peripheral"]

# вес роли в priority_score: насколько роль интересна аналитику как точка вмешательства
ROLE_WEIGHT = {
    "coordinator": 1.0,
    "consolidator": 0.9,
    "distributor": 0.8,
    "transit": 0.6,
    "terminal": 0.55,
    "peripheral": 0.1,
}

ROLE_RU = {
    "coordinator": "координатор",
    "consolidator": "консолидатор",
    "distributor": "распределитель",
    "transit": "транзит",
    "terminal": "конечный получатель",
    "peripheral": "периферия",
}
