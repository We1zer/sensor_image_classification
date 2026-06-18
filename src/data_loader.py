import os
import re
import glob
import pandas as pd

from .config import DATASET_DIR, CLASS_FOLDERS, INDEX_CSV, to_rel


def get_class_from_folder(folder_name: str) -> str | None:
    return CLASS_FOLDERS.get(folder_name)


def extract_file_number(file_path: str) -> int:
    base = os.path.basename(file_path)
    m = re.search(r"\d+", base)
    return int(m.group(0)) if m else 0


def load_image_paths(dataset_dir: str = DATASET_DIR) -> list[str]:
    paths = []
    for folder in CLASS_FOLDERS:
        full = os.path.join(dataset_dir, folder)
        if not os.path.isdir(full):
            continue
        found = set(glob.glob(os.path.join(full, "*.png"))) | \
                set(glob.glob(os.path.join(full, "*.PNG")))
        paths.extend(sorted(found))
    return paths


def create_image_index(dataset_dir: str = DATASET_DIR,
                       save_csv: bool = True) -> pd.DataFrame:
    rows = []
    for folder, cls in CLASS_FOLDERS.items():
        full = os.path.join(dataset_dir, folder)
        if not os.path.isdir(full):
            continue
        found = set(glob.glob(os.path.join(full, "*.png"))) | \
                set(glob.glob(os.path.join(full, "*.PNG")))
        files = sorted(found)
        for fp in files:
            rows.append({
                "image_path": fp,
                "class": cls,
                "file_number": extract_file_number(fp),
                "source_image": os.path.basename(fp),
                "folder": folder,
            })

    df = pd.DataFrame(rows)

    if save_csv and len(df) > 0:
        os.makedirs(os.path.dirname(INDEX_CSV), exist_ok=True)
        df_to_save = df.copy()
        df_to_save["image_path"] = df_to_save["image_path"].map(to_rel)
        df_to_save.to_csv(INDEX_CSV, index=False)

    return df


def print_dataset_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("[!] Не знайдено жодного зображення у dataset/.")
        return
    print(f"Всього зображень: {len(df)}")
    print("Розподіл за класами:")
    for cls, n in df["class"].value_counts().items():
        print(f"  {cls:10s} : {n}")


if __name__ == "__main__":
    df = create_image_index()
    print_dataset_summary(df)
