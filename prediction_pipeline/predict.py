#!/usr/bin/env python3
"""Пайплайн распознавания ГРЗ: YOLO (детекция + тип знака) -> fast-plate-ocr (чтение текста).

Источник (--source) определяется автоматически:
    каталог с изображениями            python predict.py --source images/ --output result.csv
    одно изображение                   python predict.py --source photo.jpg
    видеофайл                          python predict.py --source video.mp4 --stride 2 --save-video out.mp4
    камера (индекс)                    python predict.py --source 0 --show
    видеопоток (rtsp/http/udp/...)     python predict.py --source rtsp://host/stream --show

Веса YOLO лежат рядом с этим файлом (или в каталоге --weights-dir): best.pt
(классы: type1, type1a, type1b, other). OCR-модель fast-plate-ocr скачивается при первом запуске.

Выход: CSV (разделитель ';', UTF-8): image;plate_num;plate_type;confidence
Для видео/потока image = <имя>_f<номер кадра>, дополнительно пишется <output>_summary.csv
(уникальные номера: лучшая уверенность, первый/последний кадр, число детекций).
"""

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# Настройки
# ----------------------------------------------------------------------------
YOLO_FILE = "best.pt"
OCR_MODEL = "cct-xs-v2-global-model"
TARGET_TYPES = ("type1", "type1a", "type1b")

# Поля вокруг bbox перед OCR (доля от ширины / высоты)
PAD_X, PAD_Y = 0.04, 0.08

LAT = "ABEKMHOPCTYX"
CYR = "АВЕКМНОРСТУХ"
CYR2LAT = str.maketrans(CYR + CYR.lower(), LAT + LAT)

# Позиционные исправления частых путаний цифра <-> буква
TO_DIGIT = {"O": "0", "B": "8", "A": "4", "T": "7", "C": "0", "E": "8"}
TO_LETTER = {"0": "O", "8": "B", "4": "A", "7": "T", "5": "C"}

_L, _D = f"[{LAT}]", "[0-9]"
MASKS = {
    "type1": re.compile(f"{_L}{_D}{{3}}{_L}{{2}}(?:{_D}{{2}}|[127]{_D}{{2}})"),
    "type1a": re.compile(f"{_L}{_D}{{3}}{_L}{{2}}(?:{_D}{{2}}|[127]{_D}{{2}})"),
    "type1b": re.compile(f"{_L}{{2}}{_D}{{3}}(?:{_D}{{2}}|[127]{_D}{{2}})"),
}


# ----------------------------------------------------------------------------
# Постобработка текста
# ----------------------------------------------------------------------------
def clean(text: str, probs):
    """Верхний регистр, кириллица -> латиница, удаление не-алфавитно-цифровых символов (с выравниванием probs)."""
    pairs = [(c.upper().translate(CYR2LAT), p) for c, p in zip(text, probs)]
    pairs = [(c, p) for c, p in pairs if re.fullmatch(r"[A-Z0-9]", c)]
    return "".join(c for c, _ in pairs), [p for _, p in pairs]


def fix_by_mask(text: str, ptype: str):
    """Позиционно правит цифра<->буква (длина не меняется), проверяет маску типа.
    Возвращает строку или None, если по маске типа это не номер."""
    t = text
    if ptype == "type1b":
        kinds = "LL" + "DDD" + "D" * (len(t) - 5)
    else:
        kinds = "LDDDLL" + "D" * (len(t) - 6)
    if len(t) < 7 or len(t) > 9:
        return None
    chars = []
    for ch, k in zip(t, kinds):
        if k == "L" and ch.isdigit():
            ch = TO_LETTER.get(ch, ch)
        elif k == "D" and ch.isalpha():
            ch = TO_DIGIT.get(ch, ch)
        chars.append(ch)
    t = "".join(chars)
    return t if MASKS[ptype].fullmatch(t) else None


# ----------------------------------------------------------------------------
# Пайплайн
# ----------------------------------------------------------------------------
class Pipeline:
    def __init__(self, weights_dir: Path, device: str, imgsz: int, det_conf: float,
                 iou: float, min_conf: float, mask_thr: float):
        from ultralytics import YOLO  # импорт здесь, чтобы --help работал без зависимостей

        self.dev = torch.device(device)
        self.imgsz, self.det_conf, self.iou = imgsz, det_conf, iou
        self.min_conf, self.mask_thr = min_conf, mask_thr

        ypath = weights_dir / YOLO_FILE
        for p in (ypath,):
            if not p.is_file():
                raise FileNotFoundError(f"Не найден файл весов: {p}")
        self.yolo = YOLO(str(ypath))
        names = {int(k): str(v).lower() for k, v in self.yolo.names.items()}
        fallback = {0: "type1", 1: "type1a", 2: "type1b", 3: "other"}
        use_names = all(v in (*TARGET_TYPES, "other") for v in names.values())
        self.cls2type = names if use_names else fallback
        from fast_plate_ocr import LicensePlateRecognizer
        self.ocr = LicensePlateRecognizer(OCR_MODEL)
        self.warmup()

    def warmup(self):
        dummy = np.zeros((128, 384, 3), np.uint8)
        self._detect(dummy)
        self.ocr.run(np.zeros((64, 192, 3), dtype=np.uint8))

    def _detect(self, img):
        res = self.yolo.predict(
            img, imgsz=self.imgsz, conf=self.det_conf, iou=self.iou, device=str(self.dev),
            agnostic_nms=True, verbose=False,
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        return [(xyxy[i], float(conf[i]), self.cls2type.get(int(cls[i]), "other")) for i in range(len(cls))]

    @staticmethod
    def _crop(img, box):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = box
        px, py = (x2 - x1) * PAD_X, (y2 - y1) * PAD_Y
        x1, x2 = int(max(0, x1 - px)), int(min(w, x2 + px))
        y1, y2 = int(max(0, y1 - py)), int(min(h, y2 + py))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return img[y1:y2, x1:x2].copy()

    def process(self, img):
        """BGR-изображение -> список (plate_num, plate_type, confidence, box_xyxy)."""
        dets = [d for d in self._detect(img) if d[2] in TARGET_TYPES]
        jobs, crops = [], []
        for di, (box, _, ptype) in enumerate(dets):
            crop = self._crop(img, box)
            if crop is None:
                continue
            # fast-plate-ocr принимает целый кроп; строки типа 1А больше не дробятся.
            jobs.append(di)
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        if not crops:
            return []

        predictions = self.ocr.run(crops, return_confidence=True)
        rows = []
        for di, pred in zip(jobs, predictions):
            _, det_conf, ptype = dets[di]
            raw = pred.plate
            char_probs = pred.char_probs
            probs = list(char_probs) if char_probs is not None else []
            text, probs = clean(raw, probs if len(probs) == len(raw) else [1.0] * len(raw))
            plate = fix_by_mask(text, ptype)
            if plate is None:
                continue
            ocr_conf = float(np.mean(probs)) if probs else 1.0
            conf = float(np.sqrt(max(det_conf, 0.0) * ocr_conf))
            if conf < self.min_conf:
                continue
            if self.mask_thr > 0 and probs:
                plate = "".join(c if p >= self.mask_thr else "#" for c, p in zip(plate, probs))
            rows.append((plate, ptype, min(1.0, conf), dets[di][0]))
        return rows


IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv"}
STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://", "rtp://")


def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)  # корректно для не-ASCII путей
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def detect_source(src: str):
    """-> (kind, value): dir | image | video | stream | (None, None)."""
    s = src.strip()
    if s.isdigit():
        return "stream", int(s)  # индекс камеры
    if s.lower().startswith(STREAM_PREFIXES):
        return "stream", s
    p = Path(s)
    if p.is_dir():
        return "dir", p
    if p.is_file():
        ext = p.suffix.lower()
        if ext in IMG_EXT:
            return "image", p
        if ext in VIDEO_EXT:
            return "video", p
    return None, None


class CsvOut:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8", newline="")
        self.w = csv.writer(self.f, delimiter=";", lineterminator="\n")
        self.w.writerow(["image", "plate_num", "plate_type", "confidence"])
        self.n = 0

    def add(self, name, rows):
        for plate, ptype, conf, *_ in rows:
            self.w.writerow([name, plate, ptype, f"{conf:.3f}"])
            self.n += 1
        self.f.flush()  # результаты видны сразу, в том числе при работе с потоком

    def close(self):
        self.f.close()


def safe_process(pipe, img, name):
    try:
        return pipe.process(img)
    except Exception as e:  # один плохой кадр не должен ронять весь прогон
        print(f"[warn] ошибка на {name}: {e}", file=sys.stderr)
        return []


def draw(img, rows):
    out = img.copy()
    for plate, ptype, conf, box in rows:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{plate} {ptype} {conf:.2f}", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def run_images(pipe, files, out, verbose):
    t_total = 0.0
    for p in files:
        img = read_image(p)
        if img is None:
            print(f"[warn] не удалось прочитать {p.name}", file=sys.stderr)
            continue
        t0 = time.perf_counter()
        rows = safe_process(pipe, img, p.name)
        t_total += time.perf_counter() - t0
        out.add(p.name, rows)
        if verbose:
            for plate, ptype, conf, _ in rows:
                print(f"{p.name}: {plate} {ptype} {conf:.3f}")
    ms = 1000 * t_total / max(1, len(files))
    print(f"Готово: {len(files)} изображений, {out.n} строк; среднее время {ms:.1f} мс/кадр "
          f"(без чтения файла и записи CSV)", file=sys.stderr)


def run_capture(pipe, src, kind, args, out, name):
    """Видеофайл / камера / сетевой поток. Остановка: конец файла, --max-frames, 'q' в окне или Ctrl+C."""
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Не удалось открыть источник: {src}")
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # меньше задержка на потоках (не все бэкенды поддерживают)
    except Exception:
        pass
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 0 or fps > 240:
        fps = 25.0

    writer, show = None, args.show
    summary, last_rows = {}, []
    idx = processed = reconnects = 0
    t_total = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if kind == "stream" and reconnects < args.reconnect:
                    reconnects += 1
                    print(f"[warn] поток прерван, переподключение {reconnects}/{args.reconnect}", file=sys.stderr)
                    cap.release()
                    time.sleep(2.0)
                    cap = cv2.VideoCapture(src)
                    continue
                break
            reconnects = 0
            if args.max_frames and idx >= args.max_frames:
                break

            if idx % args.stride == 0:
                t0 = time.perf_counter()
                last_rows = safe_process(pipe, frame, f"кадр {idx}")
                t_total += time.perf_counter() - t0
                processed += 1
                out.add(f"{name}_f{idx:06d}", last_rows)
                for plate, ptype, conf, _ in last_rows:
                    d = summary.setdefault(plate, {"type": ptype, "best": conf, "first": idx, "last": idx, "n": 0})
                    d["n"] += 1
                    d["last"] = idx
                    if conf > d["best"]:
                        d["best"], d["type"] = conf, ptype

            if args.save_video or show:
                vis = draw(frame, last_rows)
                if args.save_video:
                    if writer is None:
                        h, w = vis.shape[:2]
                        writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    writer.write(vis)
                if show:
                    try:
                        cv2.imshow("predict (q - выход)", vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except cv2.error:
                        print("[warn] окно недоступно (opencv без GUI), --show отключён", file=sys.stderr)
                        show = False
            idx += 1
    except KeyboardInterrupt:
        print("\nОстановлено пользователем", file=sys.stderr)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        spath = Path(args.output).with_name(Path(args.output).stem + "_summary.csv")
        with open(spath, "w", encoding="utf-8", newline="") as f:
            wr = csv.writer(f, delimiter=";", lineterminator="\n")
            wr.writerow(["plate_num", "plate_type", "best_confidence", "first_frame", "last_frame", "detections"])
            for plate, d in sorted(summary.items(), key=lambda kv: kv[1]["first"]):
                wr.writerow([plate, d["type"], f"{d['best']:.3f}", d["first"], d["last"], d["n"]])
        ms = 1000 * t_total / max(1, processed)
        print(f"Готово: прочитано кадров {idx}, обработано {processed}, строк {out.n}, "
              f"уникальных номеров {len(summary)}; среднее время {ms:.1f} мс/кадр -> {args.output}, {spath}",
              file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", "--images", dest="source", default=os.environ.get("IMAGES_DIR"),
                    help="каталог / изображение / видеофайл / индекс камеры / URL потока (или env IMAGES_DIR)")
    ap.add_argument("--output", default=os.environ.get("OUTPUT_CSV", "result.csv"), help="выходной CSV")
    ap.add_argument("--weights-dir", default=str(HERE), help="каталог с весами (по умолчанию рядом со скриптом)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--det-conf", type=float, default=0.25, help="порог уверенности детектора")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU для NMS")
    ap.add_argument("--min-conf", type=float, default=0.0, help="отбрасывать строки с итоговой уверенностью ниже")
    ap.add_argument("--mask-thr", type=float, default=0.0, help="символы с вероятностью ниже -> '#' (0 = выкл.)")
    ap.add_argument("--stride", type=int, default=1, help="видео/поток: обрабатывать каждый N-й кадр")
    ap.add_argument("--max-frames", type=int, default=0, help="видео/поток: остановиться после N кадров (0 = без лимита)")
    ap.add_argument("--save-video", default=None, help="видео/поток: сохранить видео с разметкой (mp4)")
    ap.add_argument("--show", action="store_true", help="видео/поток: показывать окно с разметкой (q - выход)")
    ap.add_argument("--reconnect", type=int, default=5, help="поток: число попыток переподключения при обрыве")
    args = ap.parse_args()
    args.stride = max(1, args.stride)

    if not args.source:
        ap.error("укажите источник: --source PATH|URL|индекс камеры")
    kind, src = detect_source(args.source)
    if kind is None:
        ap.error(f"источник не найден или формат не поддерживается: {args.source}")

    pipe = Pipeline(Path(args.weights_dir), args.device, args.imgsz, args.det_conf,
                    args.iou, args.min_conf, args.mask_thr)
    out = CsvOut(args.output)
    try:
        if kind == "dir":
            files = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT)
            run_images(pipe, files, out, verbose=False)
        elif kind == "image":
            run_images(pipe, [src], out, verbose=True)
        else:
            name = "camera%s" % src if isinstance(src, int) else (src.stem if kind == "video" else "stream")
            run_capture(pipe, src if isinstance(src, (int, str)) else str(src), kind, args, out, name)
    finally:
        out.close()


if __name__ == "__main__":
    main()
