import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

CROPPED_DIR = os.path.join(PROCESSED_DIR, "cropped_images")
INVERTED_DIR = os.path.join(PROCESSED_DIR, "inverted_images")
PULSE_IMAGES_DIR = os.path.join(PROCESSED_DIR, "pulse_images")
PULSE_SIGNALS_DIR = os.path.join(PROCESSED_DIR, "pulse_signals")

IMAGE_PREP_DEBUG_DIR = os.path.join(PROCESSED_DIR, "image_prep_debug")

INDEX_CSV = os.path.join(FEATURES_DIR, "image_index.csv")
FEATURES_CSV = os.path.join(FEATURES_DIR, "pulse_features.csv")
PULSE_IMAGE_INDEX_CSV = os.path.join(FEATURES_DIR, "pulse_image_index.csv")


def to_rel(path: str) -> str:
    if not path:
        return path
    try:
        rel = os.path.relpath(os.path.abspath(path), BASE_DIR)
    except ValueError:
        return path
    return rel.replace("\\", "/")


def to_abs(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(BASE_DIR, path))

CLASS_FOLDERS = {
    "PNG-1acetone": "acetone",
    "PNG-1ammonia": "ammonia",
    "PNG-1ethanol": "ethanol",
    "PNG-1RH":      "RH",
}

CLASS_NAMES = ["acetone", "ammonia", "ethanol", "RH"]
NUM_CLASSES = len(CLASS_NAMES)

CROP_TOP = 35
CROP_BOTTOM = 60
CROP_LEFT = 0
CROP_RIGHT = 88

YELLOW_HSV_LOW = (15, 80, 80)
YELLOW_HSV_HIGH = (45, 255, 255)

INVERTED_THRESHOLD = 100

N_PULSES_FALLBACK = 3
PEAK_PROMINENCE_FRAC = 0.25
PEAK_MIN_DISTANCE_FRAC = 0.08
PULSE_HALF_WIDTH_FRAC = 0.12

HYST_LEVEL_PERCENTILES = (10, 90)
HYST_LOW_FRAC = 1 / 3
HYST_HIGH_FRAC = 2 / 3
HYST_MIN_DWELL_FRAC = 0.03
SEGMENT_MIN_WIDTH = 30

QC_MIN_AMPLITUDE = 15
QC_MIN_COVERAGE = 0.50
QC_MIN_WIDTH = 30
QC_MAX_WIDTH = 1200
QC_MAX_NOISE = 1.5

RANDOM_STATE = 42
TEST_SIZE = 0.25


def set_global_seeds(seed: int = RANDOM_STATE) -> None:
    import os as _os
    import random as _random
    _os.environ["PYTHONHASHSEED"] = str(seed)
    _random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except ImportError:
        pass
    try:
        import tensorflow as _tf
        _tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass

CNN_IMG_SIZE = (96, 96)

CNN_SIG_LEN = 200
CNN_SIG_SCALE = 64.0

for d in [PROCESSED_DIR, CROPPED_DIR, INVERTED_DIR, PULSE_IMAGES_DIR,
          PULSE_SIGNALS_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

IMG_SIZES = [(96, 96), (128, 128), (224, 224)]
IMG_PAD_VALUE = 0
