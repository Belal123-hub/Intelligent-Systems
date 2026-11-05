# Intelligent Systems — лабораторные работы

Студент: **Белал**

Репозиторий содержит практические работы по курсу «Интеллектуальные
системы». Работа ведётся по веткам `labN`; в `main` попадают только
проверенные и одобренные версии.

## Что где лежит

- `src/` — код всех работ (модель, обучение, поиск, анализ).
- `configs/` — YAML-конфигурации запусков.
- `data/` — фиксированное разбиение `split.json` и паспорт данных
  `manifest.json` (генерируются скриптом `src/data.py`).
- `docs/` — отчёты, прогнозы и зафиксированные результаты по работам.
- `runs/` — каталоги запусков (в git не входят, см. `.gitignore`).
- `optuna.db`, `mlflow.db` — базы Optuna и MLflow (в git не входят).

## Как воспроизвести ЛР0

1. Создать окружение и установить зависимости:
- python -m venv .venv
- venv\Scripts\activate # Windows
- pip install -r requirements.txt

2. Сгенерировать разбиение данных (один раз):
- python src/data.py

3. Запустить базовую конфигурацию (обязательно на чистом дереве Git):
- python src/train.py --config configs/base.yaml

4. Поиск гиперпараметров:
- python src/search.py

5. Честное сравнение (новые seed + test):
- python src/train.py --config configs/base.yaml --seed 101 --eval-test
- python src/train.py --config configs/best.yaml --seed 101 --eval-test
(повторить для seed 102–105)

6. Пересчитать все таблицы отчёта:
- python src/analyze_lr0.py

7. Открыть MLflow UI (опционально):
- mlflow ui --backend-store-uri sqlite:///mlflow.db

## Отчёт по ЛР0

- `docs/report_lr0.md` — итоговый отчёт на 2–3 страницы.
- `docs/prediction_lr0.md` — прогноз, закоммиченный до экспериментов.
- `docs/experiment1_runs.md`, `docs/experiment2_search.md`,
`docs/experiment3_results.md` — первичные записи экспериментов.
- `docs/experiment3_plan.md` — план Эксперимента 3, закоммиченный
до просмотра результатов на test.

## Ветки

- `main` — принятая версия.
- `lab0` — ЛР0 в работе (текущая ветка при разработке ЛР0).
- `lab1`, `lab2`, … — следующие работы (появятся по мере старта).

