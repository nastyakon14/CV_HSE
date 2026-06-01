# HW 2, 2.5

## DETR на COCO-subset и использование синтетических данных через Stable Diffusion + ControlNet

# Описание проекта

В рамках работы были выполнены два задания:

1. Fine-tuning модели DETR для задачи объектной детекции на подмножестве датасета COCO.
2. Исследование влияния синтетических данных, сгенерированных с помощью Stable Diffusion + ControlNet, на качество детектора.

Проект демонстрирует полный цикл обучения современной трансформерной модели для object detection, включая подготовку данных, обучение, логирование, профилирование, визуализацию результатов и анализ ошибок.

---

# Домашнее задание 2: DETR Object Detection

## Используемый датасет

Источник данных:

* COCO Dataset (`detection-datasets/coco`)
* Использовано 5% обучающей выборки

Для ускорения обучения были выбраны первые 10 классов COCO:

1. person
2. bicycle
3. car
4. motorcycle
5. airplane
6. bus
7. train
8. truck
9. boat
10. traffic light

После фильтрации оставлены только изображения, содержащие хотя бы один объект из выбранных классов.

Разделение данных:

* Train: 90%
* Validation: 10%

---

## Архитектура модели

Использовалась модель:

**facebook/detr-resnet-50**

Особенности DETR:

* Transformer-based object detector
* Не использует Anchor Boxes
* Выполняет прямое предсказание объектов
* Использует Bipartite Matching для сопоставления предсказаний и объектов

---

## Гиперпараметры обучения

| Параметр          | Значение          |
| ----------------- | ----------------- |
| Optimizer         | AdamW             |
| Learning Rate     | 1e-5              |
| Weight Decay      | 1e-4              |
| Batch Size        | 4                 |
| Epochs            | 3                 |
| Gradient Clipping | 0.1               |
| Scheduler         | ReduceLROnPlateau |
| Backbone          | ResNet-50         |

---

## Логирование и профилирование

Для мониторинга использовались:

### TensorBoard

Логировались:

* Total Loss
* Classification Loss
* Bounding Box Loss
* GIoU Loss
* Learning Rate
* Validation Loss
* mAP
* mAP50

Логи находятся в:

```text
runs/
```

### PyTorch Profiler

Собран trace выполнения (не загружается в github тк объем превышает 25MB):

```text
runs/detr_coco_subset/profiler/
```

Профайлер использовался для анализа:

* времени forward pass;
* времени backward pass;
* загрузки GPU;
* использования памяти.

---

## Метрики

Использовалась библиотека:

```python
torchmetrics.detection.MeanAveragePrecision
```

Рассчитывались:

* mAP
* mAP50

Результаты по эпохам сохраняются в:

```text
metrics_table.json
```

---

## Анализ ошибок

Выполнен Error Analysis следующих типов:

### Classification Error

Объект локализован корректно, но предсказан неверный класс.

### Localization Error

Класс определён правильно, однако IoU ниже порогового значения.

### False Positive

Модель обнаружила объект там, где его нет.

### False Negative

Модель не обнаружила существующий объект.

Результаты сохраняются в:

```text
error_analysis.json
```

---

## Визуализация результатов

Для валидационной выборки построены изображения с:

* зелёными GT-боксами;
* красными предсказанными боксами.

Файлы:

```text
visualizations/
```
<img width="2489" height="989" alt="Unknown-2" src="https://github.com/user-attachments/assets/5693ab57-7557-463d-9cf7-19b54c7693aa" />



---

# Домашнее задание 2.5: Stable Diffusion + ControlNet

## Цель

Исследовать влияние синтетически сгенерированных изображений на качество детектора.

---

## Поиск редких классов

Для каждого класса был подсчитан объём данных.

Редкими считались классы, входящие в нижнюю треть распределения по количеству объектов.

Визуализация распределения:

```text
visualizations/class_distribution.png
```

---

## Генерация синтетических данных

Используемые модели:

### Stable Diffusion

```text
runwayml/stable-diffusion-v1-5
```

### ControlNet

```text
lllyasviel/sd-controlnet-hed
```

### HED Detector

Использовался для построения карт границ объектов и передачи структурной информации в ControlNet.

---

## Процесс генерации

Для каждого редкого класса:

1. Выбирались реальные изображения.
2. Строилась HED-карта границ.
3. Формировался текстовый prompt.
4. Генерировалось новое изображение через Stable Diffusion + ControlNet.
5. Изображение сохранялось в папку синтетического датасета.

Структура:

```text
synthetic_data/
├── class_1/
├── class_2/
└── ...
```
<img width="1311" height="1217" alt="synthetic_samples" src="https://github.com/user-attachments/assets/74261b25-1fc7-4973-9138-fcbf5c337c5d" />



---

## Обучающие эксперименты

Были проведены два эксперимента.

### Experiment 1

Только реальные данные.

```text
baseline_real_only
```

### Experiment 2

Реальные данные + синтетические изображения.

```text
with_synthetic
```

---

## Ablation Study

Сравнивались:

* Train Loss
* Validation Loss
* mAP
* mAP50

Результаты сохраняются:

```text
ablation_table.csv
ablation_table.json
```

Графики:

```text
visualizations/ablation_curves.png
```

---

## Основные выводы

1. DETR успешно обучается на подмножестве COCO и показывает работоспособные результаты даже при небольшом количестве эпох.
2. TensorBoard позволяет отслеживать динамику всех основных компонент функции потерь.
3. Error Analysis помогает определить, связаны ли ошибки с классификацией объектов или их локализацией.
4. Stable Diffusion + ControlNet позволяют быстро расширить набор данных для редких классов.
5. Использование синтетических данных может улучшать качество модели при дефиците реальных примеров.
6. Основным ограничением данного подхода являются упрощённые псевдо-аннотации синтетических изображений.

---

# Структура репозитория

```text
.
├── hw3.ipynb
├── README.md
├── checkpoints/
├── runs/
├── visualizations/
├── synthetic_data/
├── metrics_table.json
├── error_analysis.json
├── ablation_table.csv
└── ablation_table.json
```
