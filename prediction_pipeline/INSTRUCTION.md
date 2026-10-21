# Инструкция: predict.py и evaluate.py

`predict.py` — распознавание ГРЗ (YOLO + fast-plate-ocr): изображения, видеофайл, камера, видеопоток.
`evaluate.py` — скрипт организаторов, считает метрику по результату `predict.py`.

## 1. Что нужно
- Windows 10/11 или Linux, Python 3.10–3.12 (для 3.13+ проверьте, что есть сборки torch и onnxruntime).
- Файлы рядом: `predict.py`, `requirements.txt`, веса `best.pt`; `evaluate.py`, `debug_labels.csv`, `debug_set/`.
- Интернет при первой установке и первом запуске (OCR-модель fast-plate-ocr скачивается и кэшируется).

## 2. Установка Python
- **Windows:** скачайте установщик с python.org, при установке отметьте **Add python.exe to PATH**. Проверка: `python --version`.
- **Ubuntu/Debian:** `sudo apt update && sudo apt install -y python3 python3-venv python3-pip`. Проверка: `python3 --version`.

## 3. Виртуальное окружение (в папке с predict.py)
```
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell)
source .venv/bin/activate       # Linux
python -m pip install --upgrade pip
```

## 4. Зависимости
1. **PyTorch.** Для GPU установите сборку под вашу CUDA по инструкции на pytorch.org/get-started (для старых карт вроде GTX 1050 Ti выбирайте не самую новую сборку CUDA и проверяйте поддержку архитектуры). Без GPU:
   ```
   pip install torch torchvision
   ```
2. Остальное:
   ```
   pip install -r requirements.txt
   ```
   Если pip ругается на `[onnx]`, поставьте `pip install fast-plate-ocr onnxruntime` (для GPU — `onnxruntime-gpu`).
3. Проверка:
   ```
   python -c "import torch, cv2, ultralytics, fast_plate_ocr; print(torch.cuda.is_available())"
   ```
4. После успешного запуска зафиксируйте версии: `pip freeze > requirements.lock.txt`.

## 5. Запуск predict.py
Источник определяется автоматически по `--source` (`--images` — синоним).

| Что обрабатываем | Команда |
|---|---|
| Каталог изображений | `python predict.py --source debug_set/ --output result.csv` |
| Одно изображение | `python predict.py --source photo.jpg --output one.csv` |
| Видеофайл | `python predict.py --source video.mp4 --stride 2 --save-video out.mp4 --output video.csv` |
| Веб-камера | `python predict.py --source 0 --show` |
| Видеопоток | `python predict.py --source rtsp://host:554/stream --show --output stream.csv` |

Результат: CSV `image;plate_num;plate_type;confidence` (разделитель `;`). Для видео и потока `image` = `<имя>_f<номер кадра>`, дополнительно создаётся `<output>_summary.csv` с уникальными номерами (лучшая уверенность, первый/последний кадр, число детекций). Остановка потока: `q` в окне или Ctrl+C — файлы корректно закрываются.

Параметры:

| Параметр | По умолчанию | Назначение |
|---|---|---|
| `--weights-dir` | папка скрипта | где лежит `best.pt` |
| `--device` | cuda, если есть, иначе cpu | `cpu`, `cuda`, `0` |
| `--imgsz` | 640 | размер входа YOLO (больше — лучше мелкие знаки, медленнее) |
| `--det-conf` | 0.25 | порог детектора |
| `--min-conf` | 0.0 | отбросить строки с итоговой уверенностью ниже |
| `--mask-thr` | 0.0 | символы с вероятностью ниже порога заменить на `#` |
| `--stride` | 1 | видео/поток: каждый N-й кадр |
| `--max-frames` | 0 | видео/поток: остановиться после N кадров |
| `--save-video` | — | сохранить видео с разметкой |
| `--show` | выкл. | окно с разметкой (нужен `opencv-python`, не headless) |
| `--reconnect` | 5 | попыток переподключения при обрыве потока |

Переменные окружения: `IMAGES_DIR`, `OUTPUT_CSV` (используются, если не заданы аргументы).

## 6. Оценка через evaluate.py
`evaluate.py` работает на Python 3.9+ без сторонних пакетов.
```
python predict.py --source debug_set/ --output result.csv
python evaluate.py --gt debug_labels.csv --pred result.csv --images debug_set/
python evaluate.py --gt debug_labels.csv --pred result.csv --images debug_set/ --details details.csv
```
Метрика: `Score = 0.15·A_type1 + 0.35·A_type1a + 0.35·A_type1b + 0.15·P`. Знак засчитан, если совпали все читаемые позиции и тип. `details.csv` показывает, на каких знаках теряются баллы. Если типа нет в наборе (в отладочном нет `type1b`), его вес переходит к `P`.

Что подбирать по `details.csv`: `--det-conf` (полнота), `--min-conf` (точность `P`), `--imgsz` (мелкие знаки).

## 7. Проверка датасета (validate_dataset.py)
```
python validate_dataset.py path/to/dataset
```
Создаёт `validation_report.txt`, который прикладывается к решению. Код возврата 0 — ошибок нет.

## 8. Частые проблемы
- `Не найден файл весов` — положите `best.pt` рядом со скриптом или укажите `--weights-dir`.
- `источник не найден` — проверьте путь; для URL нужен префикс `rtsp://`, `http://` и т.п.
- Ошибка при `--show` — установлен opencv без GUI: `pip uninstall opencv-python-headless && pip install opencv-python`.
- Первый запуск долгий — скачивается OCR-модель; для работы без интернета перенесите её кэш на целевую машину.
- Мало строк в CSV — знаки, не прошедшие проверку по маске номера, отбрасываются; снизьте `--det-conf`, посмотрите `details.csv`.
