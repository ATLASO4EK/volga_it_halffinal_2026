# Распознание российских ГРЗ
## Структура проекта

```
dataset/ - Итоговая версия датасета 
    generator/ - Скрипты генерации синтетических изображений 
        generate_synthetic.py
    images/
        real/
            *.png
        synthetic/
            *.png
    labels/
        *.txt
nn/ - Разработка/Дообучение Нейросетевой модели
    best_weghts/
        best.pt - yolo
        crnn_ocr_model_best.pth - custom crnn for character recognition
    data.yaml - yolo
    ocr_train.ipynb - crnn train
    train.py - yolo train
```