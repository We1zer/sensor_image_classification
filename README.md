# Sensor Image Classification

Курсовий проєкт: класифікація сенсорних сигналів за зображеннями
осцилограм методами машинного навчання.

Класи:
- `acetone`  — ацетон
- `ammonia`  — аміак
- `ethanol`  — етанол
- `RH`       — відносна вологість (контрольний/фоновий клас)

## Що робить програма

```
PNG-зображення осцилограми
        │
        ▼
  обрізання інтерфейсу осцилографа
        │
        ▼
  виділення жовтої кривої (HSV-маска)
        │
        ▼
  перетворення маски у 1D-сигнал (медіана y по стовпцях)
        │
        ▼
  згладжування + інтерполяція пропусків
        │
        ▼
  пошук імпульсів (find_peaks → fallback на 3 рівні частини)
        │
        ▼
  контроль якості імпульсу (амплітуда, ширина, шум, покриття)
        │
        ▼
  числові ознаки (амплітуда, площа, rise/fall time, енергія, ...)
        │
        ▼
  ML-моделі (LR, KNN, SVM, DT, RF, GBM)  +  опційно CNN
        │
        ▼
  голосування прогнозів імпульсів → фінальний клас + впевненість
```

## Структура проєкту

```
sensor_image_classification/
├── dataset/
│   ├── PNG-1acetone/  ← покладіть сюди свої PNG
│   ├── PNG-1ammonia/
│   ├── PNG-1ethanol/
│   └── PNG-1RH/
├── processed/        # обрізані / інвертовані / вирізані імпульси
├── features/         # CSV з ознаками
├── models/           # збережені моделі та scaler
├── results/          # графіки, confusion matrix, порівняння
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── image_preprocessing.py
│   ├── signal_extraction.py
│   ├── pulse_segmentation.py
│   ├── pulse_quality.py
│   ├── feature_extraction.py
│   ├── dataset_builder.py
│   ├── train_ml_models.py
│   ├── train_cnn.py
│   ├── evaluate.py
│   ├── predict.py
│   └── visualization.py
├── notebooks/
├── app.py
├── requirements.txt
└── README.md
```

## Швидкий старт

### 1. Встановити залежності

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

pip install -r requirements.txt
# (опційно для CNN) pip install tensorflow
```

### 2. Покласти зображення в dataset/

```
dataset/PNG-1acetone/001.png, 002.png, ...
dataset/PNG-1ammonia/001.png, 002.png, ...
dataset/PNG-1ethanol/001.png, 002.png, ...
dataset/PNG-1RH/001.png, 002.png, ...
```

Назви файлів типу `001.png`, `002.png` — програма автоматично дістає
з них `file_number` (для acetone/ammonia/ethanol номер відповідає
концентрації; для RH використовується як ID).

### 3. Запустити меню

```bash
python app.py
```

Найшвидший шлях — пункт **8 (повний pipeline)**: він зробить індексацію,
зберe ознаки, навчить моделі та побудує всі графіки.

### 3-альт. Веб-інтерфейс

Той самий функціонал доступний у браузері через Flask-додаток:

```bash
pip install -r requirements.txt
python webapp.py
# відкрити http://localhost:5000
```

Сторінки:

- `/` — Dashboard зі статусом проєкту, швидкими діями та розподілом класів.
- `/dataset` — таблиця "клас → кількість зображень", превью, переіндексація.
- `/pipeline` — побудова `pulse_features.csv` (фонова задача, прогрес-бар).
  Тут же — кнопки "Повний pipeline" і "Експеримент: original vs inverted".
- `/training` — тренування 6 ML-моделей; таблиця порівняння, confusion
  matrix, `classification_report.txt`.
- `/predict` — drag-and-drop PNG, опційні `file_number` та інверсія;
  результат показує фінальний клас, ймовірності за імпульсами і bar chart.
- `/results` — галерея PNG-графіків із `results/` (модалка з повним
  розміром), завантаження CSV/TXT.

Веб-інтерфейс не дублює логіку — він імпортує функції з `src/`
(`build_dataset`, `train_and_compare`, `predict_image`, …) та виконує
довгі задачі у фоновому потоці, опитуючи їхній статус через
`/api/job/<id>`.

### 3a. Перевірка системи без реальних даних

Якщо ви ще не маєте PNG-зображень, можна згенерувати синтетичні
осцилограми всіх 4 класів і прогнати на них pipeline:

```bash
python generate_demo_data.py --n 30   # 30 зображень × 4 класи = 120
python app.py                         # → пункт 8
```

Це створить у `dataset/PNG-1*/` файли `001.png … 030.png` із
характерними для кожного класу імпульсами і дозволить переконатись,
що все встановлено й працює. Перед роботою з реальними даними
видаліть синтетичні файли з `dataset/PNG-1*/`.

### Окремі команди (без меню)

```bash
# індексація
python -m src.data_loader

# побудова датасету ознак
python -m src.dataset_builder

# навчання та порівняння ML
python -m src.train_ml_models

# CNN (потрібен tensorflow)
python -m src.train_cnn

# оцінка
python -m src.evaluate

# прогноз для нового зображення
python -m src.predict path/to/image.png
```

## Результати, що зберігаються

Після пункту 8 у `results/` ви отримаєте:

- `class_distribution.png` — кількість імпульсів за класами
- `feature_distributions.png` — boxplot ознак
- `feature_correlation_matrix.png` — кореляція ознак
- `signal_extraction_example.png` — приклад: оригінал → маска → сигнал
- `examples_of_detected_pulses.png` — знайдені межі імпульсів
- `concentration_analysis_amplitude.png` — амплітуда vs file_number
- `concentration_analysis_area.png` — площа vs file_number
- `model_comparison.png` + `model_comparison.csv` — порівняння моделей
- `confusion_matrix_<best>.png` — матриця помилок найкращої моделі
- `classification_report.txt` — точність по класах

У `models/`:
- `best_ml_model.pkl`, `scaler.pkl`, `label_encoder.pkl`,
  `best_model_meta.json` — все необхідне для прогнозу.

## Ключові методичні рішення

1. **Group train/test split за `source_image`** — імпульси одного
   зображення не потрапляють і в train, і в test. Це уникає
   завищеної точності.
2. **Голосування за імпульсами** — фінальний клас зображення
   формується з усереднених імовірностей якісних імпульсів.
3. **Контроль якості імпульсів** — фільтрація за амплітудою, шириною,
   покриттям маски, рівнем шуму. Погані імпульси виключаються.
4. **Експеримент original vs inverted** (пункт 9 меню) — окремо тренує
   моделі на оригінальних і на інвертованих зображеннях, порівнює
   результати. Це окремий розділ для курсової.
5. **Аналіз концентрації** — для acetone/ammonia/ethanol будується
   залежність ознак від `file_number`. Для RH цей зв'язок не
   передбачається (контрольний клас).

## Розширене порівняння моделей

`src/train_ml_models.py` `train_and_compare()` підтримує три режими
понад базові 6 моделей:

- **XGBoost** (`use_xgboost=True`, авто, якщо встановлений) — 7-ма
  базова модель, сучасний градієнтний бустинг.
- **Voting (soft) + Stacking** (`use_ensembles=True`) — ансамблі
  будуються з top-3 базових моделей за `f1_macro`. Stacking
  використовує `LogisticRegression` як фінальний оцінювач.
- **GridSearchCV для top-k** (`tune_top_k=2`) — підбір
  гіперпараметрів через `GridSearchCV` з `GroupKFold(n=3)` за
  `source_image`. Тюнені моделі додаються у таблицю порівняння як
  `<name>_tuned`, найкращі параметри зберігаються в
  `model_comparison.csv` колонка `best_params`.

У веб-інтерфейсі (`/training`) усі три режими виносяться окремими
чекбоксами; у таблиці результатів кожен рядок має тип:
`base` / `tuned` / `ensemble`.

Програмно:

```python
from src.train_ml_models import train_and_compare

res = train_and_compare(
    use_xgboost=True,
    use_ensembles=True,
    tune_top_k=2,
)
```

Сітки гіперпараметрів задані у `PARAM_GRIDS` (тому ж файлі) — їх
можна звузити чи розширити під свої експерименти.

## Спектральні (FFT) ознаки

`src/feature_extraction.py` додає до 19 часо-просторових ознак ще
9 спектральних (через `scipy.fft.rfft`):

- `fft_centroid` — спектральний центроїд (середня частота за енергією);
- `fft_bandwidth` — спектральна ширина;
- `fft_rolloff` — частота, нижче якої 85% енергії;
- `fft_ratio_low / mid / high` — частки енергії в трьох смугах;
- `fft_peak_freq`, `fft_peak_mag` — частота і амплітуда головного спектрального піка;
- `fft_flatness` — Wiener entropy (тон vs шум).

Ці ознаки часто допомагають розрізняти схожі органічні сполуки
(наприклад acetone vs ethanol), де часові ознаки сильно перетинаються.

## Signal augmentation

`src/signal_augmentation.py` + параметр `augment_factor` у
`build_dataset()` дозволяють згенерувати N додаткових копій
кожного валідного імпульсу з:

- гауссовим шумом ~3% від std;
- амплітудним скейлом у [0.90; 1.10];
- циклічним часовим зсувом до ±4% довжини;
- (з ймовірністю 50%) невеликим time-warp.

Аугментовані рядки помічаються `is_augmented=True` у CSV.
**Важливо**: `source_image` зберігається тим самим, тому
`GroupShuffleSplit` гарантує, що оригінал та його аугментовані копії
не потрапляють одночасно в train і test.

**Train-only augmentation policy.** Після group-split і `train_ml_models.py`,
і `evaluate.py` додатково фільтрують `test_idx`, залишаючи лише рядки
з `is_augmented=False`. Тобто:

- **Train** отримує усі дані (оригінали + аугментовані копії) → більше зразків для навчання.
- **Test** містить виключно реальні (неаугментовані) приклади → метрики чесні щодо справжніх сенсорних даних, а не синтетично спотворених копій.

Це стандартна практика наукових публікацій з augmentation.

У веб-інтерфейсі параметр `augment_factor` виставляється на сторінці
`/pipeline` (число копій на імпульс, 0 = вимкнено). Рекомендований
діапазон 2–3.

## Ансамблі: як виправлено Stacking

Початкова версія Stacking використовувала тільки top-3 базові моделі —
вони часто корельовані (RF + GBM + XGB) і дають однакові помилки, що
позбавляє ансамбль сенсу. Виправлено:

- Базові — **усі** доступні моделі (різноманітність помилок);
- `passthrough=True` — фінальна `LogisticRegression` бачить і вихідні
  ознаки, не лише ймовірності base-моделей;
- `cv=5` (адаптивно зменшується, якщо мало груп);
- `n_jobs=1` для StackingClassifier — уникаємо вкладеного паралелізму
  з `n_jobs=-1` всередині base-моделей.

Voting (soft) залишається на top-3 — його сенс саме в усередненні
найкращих, не в різноманітності.

## Підлаштування під ваші зображення

Якщо інтерфейс осцилографа на ваших скріншотах інший, відкоригуйте
константи в `src/config.py`:

```python
CROP_TOP = 35      # пікселів зверху (логотип/час)
CROP_BOTTOM = 60   # знизу (меню)
CROP_RIGHT = 200   # справа (панель параметрів)

YELLOW_HSV_LOW  = (15, 80, 80)
YELLOW_HSV_HIGH = (45, 255, 255)
```

Запустіть `cmd_plots` (пункт 7) — побачите, наскільки чисто
виділяється сигнал. Якщо краї графіка обрізає неправильно —
поправте CROP_*.
