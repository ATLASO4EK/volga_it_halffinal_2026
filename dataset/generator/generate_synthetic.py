from pathlib import Path
from io import BytesIO
import random

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

GENERATOR_DIR = Path(__file__).resolve().parent
DATASET_DIR = GENERATOR_DIR.parent

REAL_DIR = DATASET_DIR / "images" / "real"
SYNTHETIC_DIR = DATASET_DIR / "images" / "synthetic"
LABELS_DIR = DATASET_DIR / "labels"
META_CSV = DATASET_DIR / "meta.csv"

IMAGE_EXTENSIONS = { ".jpg" }

RANDOM_SEED = 33
random.seed(RANDOM_SEED)


def change_hue(image, shift):
    """Сдвигает оттенок изображения на заданное число единиц HSV (0–255)."""
    has_alpha = image.mode == "RGBA"
    rgba = image.convert("RGBA") if has_alpha else None

    rgb = image.convert("RGB")
    hsv = rgb.convert("HSV")

    h, s, v = hsv.split()
    h = h.point(lambda value: (value + shift) % 256)

    result = Image.merge("HSV", (h, s, v)).convert("RGB")

    if rgba is not None:
        result.putalpha(rgba.getchannel("A"))

    return result


def add_grain(image, intensity=18):
    """Добавляет мелкозернистый шум, не меняя размеры изображения."""
    original_mode = image.mode
    rgba = image.convert("RGBA")
    rgb = rgba.convert("RGB")

    # Создаём случайную зернистую текстуру
    noise = Image.effect_noise(image.size, 100).convert("L")

    # Центрируем шум относительно серого и регулируем интенсивность
    noise = noise.point(
        lambda value: max(
            0, min(255, int(128 + (value - 128) * intensity / 35))
        )
    )

    # Смешивание с оригиналом создаёт эффект плёночного зерна
    gray = Image.merge("RGB", (noise, noise, noise))
    result = Image.blend(rgb, gray, 0.12)

    if original_mode == "RGBA":
        result.putalpha(rgba.getchannel("A"))

    return result


def jpeg_artifacts(image, quality=45):
    """Имитирует артефакты сжатия JPEG."""
    buffer = BytesIO()

    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=quality
    )

    buffer.seek(0)

    with Image.open(buffer) as compressed:
        result = compressed.convert("RGB").copy()

    if image.mode == "RGBA":
        result.putalpha(image.getchannel("A"))

    return result


def apply_augmentations(image):
    """Возвращает ровно 9 различных преобразований."""

    results = []

    # 1. Повышенная яркость
    results.append(
        ImageEnhance.Brightness(image).enhance(1.25)
    )

    # 2. Пониженная яркость
    results.append(
        ImageEnhance.Brightness(image).enhance(0.75)
    )

    # 3. Повышенный контраст
    results.append(
        ImageEnhance.Contrast(image).enhance(1.35)
    )

    # 4. Пониженный контраст
    results.append(
        ImageEnhance.Contrast(image).enhance(0.75)
    )

    # 5. Повышенная насыщенность
    results.append(
        ImageEnhance.Color(image).enhance(1.5)
    )

    # 6. Сдвиг оттенка
    results.append(
        change_hue(image, shift=12)
    )

    # 7. Повышенная резкость
    results.append(
        ImageEnhance.Sharpness(image).enhance(2.0)
    )

    # 8. Лёгкое размытие
    results.append(
        image.filter(ImageFilter.GaussianBlur(radius=1.0))
    )

    # 9. Шум и сжатие JPEG
    results.append(
        add_grain(jpeg_artifacts(image, quality=65))
    )

    return results


def save_image(image, output_path):
    """Сохраняет изображение, учитывая формат файла и прозрачность."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, "white")
            background.paste(image, mask=image.getchannel("A"))
            image = background
        else:
            image = image.convert("RGB")

    elif suffix == ".png":
        if image.mode not in {"RGB", "RGBA", "L", "LA", "P"}:
            image = image.convert("RGBA")

    elif suffix in {".tif", ".tiff"}:
        if image.mode not in {"RGB", "RGBA", "L", "I", "F"}:
            image = image.convert("RGB")

    elif suffix in {".bmp", ".webp"}:
        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGB")

    image.save(output_path)


def main():
    if not REAL_DIR.exists():
        raise FileNotFoundError(
            f"Не найдена папка с оригиналами: {REAL_DIR}"
        )

    files = sorted(
        path for path in REAL_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not files:
        print(f"В папке {REAL_DIR} нет изображений.")
        return

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    total_generated = 0
    errors = 0

    print(f"Найдено исходных изображений: {len(files)}")
    print(f"Папка результата: {SYNTHETIC_DIR}")
    print()

    for index, source_path in enumerate(files, start=1):
        try:
            with Image.open(source_path) as original:
                # Применяем EXIF-ориентацию, чтобы сохранить
                # правильное расположение объектов на фотографии.
                image = ImageOps.exif_transpose(original).copy()

            # Сохраняем структуру вложенных папок real/
            relative_path = source_path.relative_to(REAL_DIR)
            relative_parent = relative_path.parent
            stem = source_path.stem

            output_dir = SYNTHETIC_DIR / relative_parent
            output_dir.mkdir(parents=True, exist_ok=True)

            augmented_images = apply_augmentations(image)

            for aug_index, augmented in enumerate(
                augmented_images, start=1
            ):
                output_path = (
                    output_dir /
                    f"{stem}_{aug_index:02d}{source_path.suffix.lower()}"
                )

                save_image(augmented, output_path)
                total_generated += 1

            print(
                f"[{index}/{len(files)}] {source_path.name}: "
                f"создано {len(augmented_images)} изображений"
            )

        except Exception as error:
            errors += 1
            print(f"[ОШИБКА] {source_path}: {error}")

    print("\nГенерация завершена.")
    print(f"Исходных изображений: {len(files)}")
    print(f"Создано синтетических: {total_generated}")
    print(f"Ошибок обработки: {errors}")
    print(f"Результат: {SYNTHETIC_DIR}")


if __name__ == "__main__":
    main()
