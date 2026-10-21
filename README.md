# Распознавание нестандартных ГРЗ (типы 1, 1А, 1Б и негативные примеры)

Волга-IT 2026, полуфинал «ИИ и анализ данных».

Решение находит на изображении, видеофайле или видеопотоке государственные регистрационные знаки, определяет их тип (`type1`, `type1a`, `type1b`, `other`) и читает номер. Результат — CSV `image;plate_num;plate_type;confidence`.

## Как это работает

```
кадр -> YOLO (детекция + тип знака) -> кроп -> fast-plate-ocr (текст) -> проверка по маске номера -> CSV
```

1. **YOLO (Ultralytics)** — находит знаки и классифицирует тип. Дообучена на собственном датасете (`nn/`).
2. **fast-plate-ocr** (`cct-xs-v2-global-model`) — читает текст с кропа целиком, включая двухстрочные знаки 1А.
3. **Постобработка** — перевод кириллицы в латиницу, позиционное исправление цифра↔буква, проверка по маске `Л ЦЦЦ ЛЛ РЕГИОН` (для 1Б — `ЛЛ ЦЦЦ РЕГИОН`). Номера, не прошедшие проверку, не выводятся.

## Структура репозитория

| Каталог | Содержимое |
|---|---|
| `dataset/` | Итоговый датасет: `images/real`, `images/synthetic`, `labels/` (YOLO), `meta.csv`, `generator/`, `LICENSE`, датащит `README.md`, отчёт валидатора (`report.txt`, `report.json`) |
| `nn/` | Обучение детектора: `train.py`, `data.yaml`, `yolo_format_dataset/`, `best_weghts/best.pt`, `requirements.txt` |
| `prediction_pipeline/` | Готовый пайплайн: `predict.py`, `INSTRUCTION.md`, `requirements.txt`, `best.pt`, материалы организаторов (`evaluate.py`, `debug_set/`, `debug_labels.csv`) и результаты (`result.csv`, `details.csv`) |

## Быстрый старт

```
cd prediction_pipeline
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
# установите PyTorch под вашу видеокарту (pytorch.org), затем:
pip install -r requirements.txt
python predict.py --source <каталог | изображение | видео | 0 | rtsp://...> --output result.csv
```

Полная инструкция (установка Python, параметры, видео и потоки, оценка): [`prediction_pipeline/INSTRUCTION.md`](prediction_pipeline/INSTRUCTION.md).

Оценка на отладочном наборе:

```
python evaluate.py --gt debug_labels.csv --pred result.csv --images debug_set/ --details details.csv
```

## Датасет

Реальные и синтетические изображения, разметка в формате YOLO и `meta.csv` с источниками и лицензиями. Подробности: [`dataset/README.md`](dataset/README.md). Синтетика воспроизводится скриптом `dataset/generator/generate_synthetic.py` (фиксированный seed, зависимости в `dataset/generator/requirements.txt`). Датасет опубликован под CC BY 4.0.

## Ограничения

- Детектор классифицирует тип знака, поэтому ошибки типа приводят к потере знака (тип входит в критерий).
- Знаки, не прошедшие проверку по маске номера, отбрасываются (повышает точность, снижает полноту).
- Задания по знакам вне транспортных средств (`is_vehicle=0`) решаются только через класс `other`.
- <добавьте наблюдаемые слабые места: ночь, мелкие знаки, сильный наклон и т. д.>

## Лицензии

- Датасет: CC BY 4.0 (`dataset/LICENSE`).
- Код: <лицензия, см. `prediction_pipeline/LICENSE`>.
- Используемые компоненты: Ultralytics YOLO (AGPL-3.0), fast-plate-ocr и его предобученные модели (проверьте лицензию на странице проекта).
- Материалы организаторов (`debug_set/`, `debug_labels.csv`, `evaluate.py`) используются только для отладки и в датасет не входят.
