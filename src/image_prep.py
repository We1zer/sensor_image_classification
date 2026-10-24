import os
import glob
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from . import config as cfg
from .image_preprocessing import extract_yellow_mask, denoise_mask


def scan_pulse_images(root: str = cfg.PULSE_IMAGES_DIR,
                      save_csv: bool = True) -> pd.DataFrame:
    rows = []
    for folder, cls in cfg.CLASS_FOLDERS.items():
        full = os.path.join(root, folder)
        if not os.path.isdir(full):
            continue
        found = set(glob.glob(os.path.join(full, "*.png"))) | \
                set(glob.glob(os.path.join(full, "*.PNG")))
        for fp in sorted(found):
            stem = os.path.splitext(os.path.basename(fp))[0]
            parts = stem.split("__")
            if len(parts) != 3:
                continue
            _, source_stem, pulse_tag = parts
            try:
                pulse_index = int(pulse_tag.lstrip("p"))
            except ValueError:
                pulse_index = 0
            rows.append({
                "pulse_image_path": fp,
                "class": cls,
                "folder": folder,
                "source_image": f"{source_stem}.png",
                "image_key": f"{cls}/{source_stem}",
                "pulse_index": pulse_index,
            })

    df = pd.DataFrame(rows)
    if save_csv and not df.empty:
        os.makedirs(os.path.dirname(cfg.PULSE_IMAGE_INDEX_CSV), exist_ok=True)
        df_save = df.copy()
        df_save["pulse_image_path"] = df_save["pulse_image_path"].map(cfg.to_rel)
        df_save.to_csv(cfg.PULSE_IMAGE_INDEX_CSV, index=False)
    return df


@dataclass(frozen=True)
class PrepSpec:
    resize_mode: str = "letterbox"
    channels: str = "gray"
    size: tuple[int, int] = (96, 96)
    denoise: bool = True
    pad_value: int = cfg.IMG_PAD_VALUE
    clean_overlay: bool = False

    @property
    def name(self) -> str:
        h, w = self.size
        return (f"{self.resize_mode}_{self.channels.replace('+', '-')}_{h}x{w}"
                + ("_clean" if self.clean_overlay else ""))

    @property
    def n_channels(self) -> int:
        return {"gray": 1, "mask": 1, "gray+mask": 2, "rgb": 3}[self.channels]


SPEC_BASELINE = PrepSpec(resize_mode="stretch", channels="gray", size=(96, 96))

SPEC_MAIN = PrepSpec(resize_mode="letterbox", channels="gray+mask",
                     size=(192, 96), clean_overlay=True)


def letterbox_resize(img: np.ndarray, size: tuple[int, int],
                     pad_value: int = 0) -> np.ndarray:
    h, w = size
    ih, iw = img.shape[:2]
    scale = min(h / ih, w / iw)
    nh, nw = max(1, int(round(ih * scale))), max(1, int(round(iw * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)

    top = (h - nh) // 2
    bottom = h - nh - top
    left = (w - nw) // 2
    right = w - nw - left
    if top or bottom or left or right:
        resized = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=pad_value)
    return resized


def stretch_resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def overlay_box_mask(img_bgr: np.ndarray) -> np.ndarray:
    i = img_bgr.astype(np.int16)
    b, g, r = i[..., 0], i[..., 1], i[..., 2]
    grey = ((np.abs(b - g) < 10) & (np.abs(g - r) < 10)
            & (b >= 35) & (b <= 100)).astype(np.uint8)
    solid = cv2.morphologyEx(grey, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((9, 25), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(solid)
    box = np.zeros(grey.shape, dtype=bool)
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area >= 100 and h >= 12 and w >= 5:
            box[max(0, y - 2):y + h + 2, max(0, x - 2):x + w + 2] = True
    return box


def build_channels(img_bgr: np.ndarray, spec: PrepSpec) -> tuple[np.ndarray, dict]:
    stages: dict[str, np.ndarray] = {"original": img_bgr}

    if spec.clean_overlay:
        box = overlay_box_mask(img_bgr)
        if box.any():
            img_bgr = img_bgr.copy()
            img_bgr[box] = 0
            stages["overlay_box"] = box.astype(np.uint8) * 255

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = None
    if "mask" in spec.channels:
        mask = extract_yellow_mask(img_bgr)
        if spec.denoise:
            mask = denoise_mask(mask, kernel_size=3)
        stages["mask"] = mask

    if spec.channels == "gray":
        out = gray[:, :, None]
    elif spec.channels == "mask":
        out = mask[:, :, None]
    elif spec.channels == "gray+mask":
        out = np.stack([gray, mask], axis=-1)
    elif spec.channels == "rgb":
        out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Невідомий channels: {spec.channels}")

    stages["channels_full"] = out
    return out, stages


def prepare_one(path: str, spec: PrepSpec,
                return_stages: bool = False):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return (None, {}) if return_stages else None
    return prepare_array(img, spec, return_stages=return_stages)


def prepare_array(img_bgr: np.ndarray, spec: PrepSpec,
                  return_stages: bool = False):
    chans, stages = build_channels(img_bgr, spec)

    if spec.resize_mode == "letterbox":
        resized = letterbox_resize(chans, spec.size, spec.pad_value)
    elif spec.resize_mode == "stretch":
        resized = stretch_resize(chans, spec.size)
    else:
        raise ValueError(f"Невідомий resize_mode: {spec.resize_mode}")

    if resized.ndim == 2:
        resized = resized[:, :, None]

    stages["resized"] = resized
    out = resized.astype(np.float32) / 255.0
    return (out, stages) if return_stages else out


@dataclass
class ImageData:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    image_key: np.ndarray
    pulse_id: np.ndarray
    class_names: list
    spec: "PrepSpec"

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def build_image_array(df: pd.DataFrame | None = None,
                      spec: PrepSpec = PrepSpec(),
                      dump_debug: bool = False,
                      dump_per_class: int = 8,
                      verbose: bool = True) -> ImageData:
    if df is None:
        df = scan_pulse_images()
    if df.empty:
        raise ValueError(
            "Не знайдено вирізаних імпульсів у processed/pulse_images/. "
            "Спочатку побудуйте їх: python -m src.dataset_builder")

    class_names = sorted(df["class"].unique().tolist())
    c2i = {c: i for i, c in enumerate(class_names)}

    dump_left = {c: dump_per_class for c in class_names} if dump_debug else {}
    dump_root = os.path.join(cfg.IMAGE_PREP_DEBUG_DIR, spec.name)

    Xs, ys, groups, keys, pids = [], [], [], [], []
    n_failed = 0
    for _, row in df.iterrows():
        path = cfg.to_abs(row["pulse_image_path"])
        need_dump = dump_debug and dump_left.get(row["class"], 0) > 0
        res = prepare_one(path, spec, return_stages=need_dump)
        if need_dump:
            arr, stages = res
        else:
            arr, stages = res, None
        if arr is None:
            n_failed += 1
            continue

        if need_dump:
            _dump_stages(stages, spec, dump_root, row)
            dump_left[row["class"]] -= 1

        Xs.append(arr)
        ys.append(c2i[row["class"]])
        groups.append(row["source_image"])
        keys.append(row["image_key"])
        pids.append(os.path.splitext(os.path.basename(str(
            row["pulse_image_path"])))[0])

    if not Xs:
        raise ValueError("Не вдалось завантажити жодного зображення імпульсу.")

    data = ImageData(
        X=np.stack(Xs),
        y=np.array(ys, dtype=np.int64),
        groups=np.array(groups),
        image_key=np.array(keys),
        pulse_id=np.array(pids),
        class_names=class_names,
        spec=spec,
    )

    if verbose:
        print(f"  [{spec.name}] X={data.X.shape}, класів={len(class_names)}, "
              f"груп={len(np.unique(data.groups))}, "
              f"знімків={len(np.unique(data.image_key))}"
              + (f", не прочитано: {n_failed}" if n_failed else ""))
        if dump_debug:
            print(f"  [debug] проміжні картинки: {cfg.to_rel(dump_root)}/")

    return data


def _to_viewable(arr: np.ndarray) -> np.ndarray:
    a = arr
    if a.dtype != np.uint8:
        a = np.clip(a * 255.0, 0, 255).astype(np.uint8) if a.max() <= 1.0 \
            else np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        return a
    c = a.shape[2]
    if c == 1:
        return a[:, :, 0]
    if c == 2:
        gray, mask = a[:, :, 0], a[:, :, 1]
        return np.stack([gray, mask, gray], axis=-1)
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)


def _imwrite(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(path)


def _dump_stages(stages: dict, spec: PrepSpec, dump_root: str,
                 row: pd.Series) -> None:
    stem = os.path.splitext(os.path.basename(str(row["pulse_image_path"])))[0]
    out_dir = os.path.join(dump_root, str(row["class"]), stem)

    original = stages["original"]
    _imwrite(os.path.join(out_dir, "0_original.png"), original)

    if "mask" in stages:
        _imwrite(os.path.join(out_dir, "1_mask.png"), stages["mask"])

    final = _to_viewable(stages["resized"])
    _imwrite(os.path.join(out_dir, "2_input.png"), final)

    big = cv2.resize(final, (final.shape[1] * 6, final.shape[0] * 6),
                     interpolation=cv2.INTER_NEAREST)
    _imwrite(os.path.join(out_dir, "3_input_x6.png"), big)

    _imwrite(os.path.join(out_dir, "4_side_by_side.png"),
             _side_by_side(original, big))


def _side_by_side(left: np.ndarray, right: np.ndarray,
                  gap: int = 12) -> np.ndarray:
    def _bgr(a):
        return a if a.ndim == 3 else cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)

    l, r = _bgr(left), _bgr(right)
    h = max(l.shape[0], r.shape[0])

    def _pad_to(a):
        if a.shape[0] == h:
            return a
        top = (h - a.shape[0]) // 2
        return cv2.copyMakeBorder(a, top, h - a.shape[0] - top, 0, 0,
                                  cv2.BORDER_CONSTANT, value=(40, 40, 40))

    sep = np.full((h, gap, 3), 90, dtype=np.uint8)
    return np.hstack([_pad_to(l), sep, _pad_to(r)])


def dump_debug_images(specs: list[PrepSpec] | None = None,
                      per_class: int = 8,
                      montage: bool = True,
                      verbose: bool = True) -> list[str]:
    if specs is None:
        specs = default_specs()

    df = scan_pulse_images()
    if df.empty:
        raise ValueError(
            "Немає вирізаних імпульсів. Запустіть: python -m src.dataset_builder")

    made = []
    for spec in specs:
        if verbose:
            print(f"[*] {spec.name}")
        data = build_image_array(df, spec, dump_debug=True,
                                 dump_per_class=per_class, verbose=verbose)
        dump_root = os.path.join(cfg.IMAGE_PREP_DEBUG_DIR, spec.name)
        if montage:
            _save_montage(data.X, data.y, data.class_names, spec,
                          dump_root, per_class)
        made.append(dump_root)

    if verbose:
        print()
        print(f"[OK] Дамп готовий: {cfg.to_rel(cfg.IMAGE_PREP_DEBUG_DIR)}/")
        print("     Дивіться montage.png у папці кожної специфікації, "
              "далі — <клас>/<імпульс>/4_side_by_side.png")
    return made


def _save_montage(X: np.ndarray, y: np.ndarray, class_names: list[str],
                  spec: PrepSpec, dump_root: str, per_class: int) -> None:
    cell_h, cell_w = spec.size
    scale = max(1, 220 // max(cell_h, cell_w))
    ch, cw = cell_h * scale, cell_w * scale
    pad, label_w = 6, 110

    rows = []
    for ci, cname in enumerate(class_names):
        idx = np.flatnonzero(y == ci)[:per_class]
        cells = []
        for i in idx:
            img = _to_viewable(X[i])
            img = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_NEAREST)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                     cv2.BORDER_CONSTANT, value=(35, 35, 35))
            cells.append(img)
        while len(cells) < per_class:
            cells.append(np.full((ch + 2 * pad, cw + 2 * pad, 3), 35, np.uint8))

        strip = np.hstack(cells)
        label = np.full((strip.shape[0], label_w, 3), 35, dtype=np.uint8)
        cv2.putText(label, cname, (8, strip.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
        rows.append(np.hstack([label, strip]))

    grid = np.vstack(rows)
    title = np.full((36, grid.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(title, spec.name, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 1, cv2.LINE_AA)
    _imwrite(os.path.join(dump_root, "montage.png"), np.vstack([title, grid]))


def default_specs() -> list[PrepSpec]:
    return [
        SPEC_BASELINE,
        PrepSpec("letterbox", "gray", (96, 96)),
        PrepSpec("letterbox", "rgb", (96, 96)),
        PrepSpec("letterbox", "mask", (96, 96)),
        PrepSpec("letterbox", "gray+mask", (96, 96)),
        PrepSpec("letterbox", "gray+mask", (128, 128)),
        PrepSpec("letterbox", "gray+mask", (192, 96)),
        PrepSpec("letterbox", "gray+mask", (256, 128)),
    ]


if __name__ == "__main__":
    dump_debug_images()
