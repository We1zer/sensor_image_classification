import os
import sys
import glob
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config as cfg
from src.data_loader import create_image_index, print_dataset_summary
from src.dataset_builder import build_dataset
from src.train_ml_models import train_and_compare
from src.evaluate import evaluate_saved_model
from src.predict import predict_image
from src.visualization import (
    plot_pipeline_example, plot_pulse_segmentation_example,
    plot_class_distribution, plot_feature_distributions,
    plot_feature_correlation, plot_confusion, plot_model_comparison,
    plot_concentration_analysis, generate_all_plots,
)
import pandas as pd


MENU = """
============================================================
   Sensor Image Classification System
============================================================
1. Створити індекс зображень (dataset/)
2. Побудувати датасет ознак (preprocessing → pulses → features)
3. Навчити ML-моделі та порівняти
4. Навчити CNN-модель (опційно)
5. Оцінити збережену ML-модель
6. Класифікувати нове зображення
7. Побудувати графіки дослідження
8. Запустити повний pipeline (1 → 2 → 3 → 7)
9. Експеримент: original vs inverted (порівняння)
0. Вийти
------------------------------------------------------------
Ваш вибір: """


def _find_first_image() -> str | None:
    paths = []
    for folder in cfg.CLASS_FOLDERS:
        paths.extend(sorted(glob.glob(
            os.path.join(cfg.DATASET_DIR, folder, "*.png"))))
    return paths[0] if paths else None


def cmd_index() -> None:
    df = create_image_index()
    print_dataset_summary(df)
    print(f"Збережено індекс: {cfg.INDEX_CSV}")


def cmd_build_dataset() -> None:
    inv = input("Інвертувати кольори зображень? (y/N): ").strip().lower() == "y"
    build_dataset(invert=inv, save_pulse_images=True, save_pulse_signals=False)


def cmd_train_ml() -> None:
    use_fn = input("Використати file_number як ознаку? (y/N): ").strip().lower() == "y"
    train_and_compare(use_file_number=use_fn)


def cmd_train_cnn() -> None:
    try:
        from src.train_cnn import train_cnn
    except ImportError as e:
        print(f"[!] {e}")
        return
    epochs = input("Кількість епох [15]: ").strip() or "15"
    try:
        train_cnn(epochs=int(epochs))
    except Exception as e:
        print(f"[!] Помилка тренування CNN: {e}")
        traceback.print_exc()


def cmd_evaluate() -> None:
    try:
        evaluate_saved_model()
    except FileNotFoundError as e:
        print(f"[!] {e}")


def cmd_predict() -> None:
    path = input("Шлях до PNG-зображення: ").strip().strip('"').strip("'")
    if not path:
        return
    try:
        predict_image(path)
    except Exception as e:
        print(f"[!] Помилка прогнозу: {e}")
        traceback.print_exc()


def cmd_plots() -> None:
    if not os.path.exists(cfg.FEATURES_CSV):
        print(f"[!] Немає {cfg.FEATURES_CSV}. Спочатку побудуйте датасет (п.2).")
        return
    df = pd.read_csv(cfg.FEATURES_CSV)
    example = _find_first_image()
    res = None
    try:
        res = evaluate_saved_model(verbose=False)
    except Exception:
        pass
    saved = generate_all_plots(df, results_dict={
        "results": pd.read_csv(os.path.join(cfg.RESULTS_DIR, "model_comparison.csv")),
        "best_name": res["best_name"] if res else "model",
        "confusion": res["confusion"] if res else None,
        "class_names": res["class_names"] if res else None,
    } if res else None,
    example_image=example)
    print(f"[OK] Збережено графіків: {len(saved)}")
    for s in saved:
        print(f"   - {s}")


def cmd_full() -> None:
    cmd_index()
    print()
    print("[*] Будуємо датасет...")
    df = build_dataset(invert=False, save_pulse_images=True)
    if df.empty:
        return
    print()
    print("[*] Тренуємо ML-моделі...")
    res = train_and_compare()
    print()
    print("[*] Будуємо графіки...")
    example = _find_first_image()
    generate_all_plots(df, results_dict=res, example_image=example)
    print("[OK] Повний pipeline завершено.")


def cmd_experiment_invert() -> None:
    print("[*] Збираємо датасет на оригінальних зображеннях...")
    df_o = build_dataset(invert=False, save_pulse_images=False, verbose=False)
    if df_o.empty:
        print("[!] Не вдалось зібрати original-датасет.")
        return
    df_o.to_csv(os.path.join(cfg.FEATURES_DIR, "pulse_features_original.csv"),
                index=False)
    res_o = train_and_compare(features_csv=os.path.join(
        cfg.FEATURES_DIR, "pulse_features_original.csv"), verbose=False)

    print("[*] Збираємо датасет на інвертованих зображеннях...")
    df_i = build_dataset(invert=True, save_pulse_images=False, verbose=False)
    if df_i.empty:
        print("[!] Не вдалось зібрати inverted-датасет.")
        return
    df_i.to_csv(os.path.join(cfg.FEATURES_DIR, "pulse_features_inverted.csv"),
                index=False)
    res_i = train_and_compare(features_csv=os.path.join(
        cfg.FEATURES_DIR, "pulse_features_inverted.csv"), verbose=False)

    print()
    print("=== Порівняння (за найкращою моделлю) ===")
    print(f"Original  best={res_o['best_name']:18s} f1={res_o['results'].iloc[0]['f1_macro']:.4f}")
    print(f"Inverted  best={res_i['best_name']:18s} f1={res_i['results'].iloc[0]['f1_macro']:.4f}")

    out = os.path.join(cfg.RESULTS_DIR, "original_vs_inverted.csv")
    pd.DataFrame([
        {"variant": "original",
         "best_model": res_o["best_name"],
         "f1_macro": res_o["results"].iloc[0]["f1_macro"],
         "accuracy": res_o["results"].iloc[0]["accuracy"]},
        {"variant": "inverted",
         "best_model": res_i["best_name"],
         "f1_macro": res_i["results"].iloc[0]["f1_macro"],
         "accuracy": res_i["results"].iloc[0]["accuracy"]},
    ]).to_csv(out, index=False)
    print(f"[OK] Результат: {out}")


def main() -> None:
    actions = {
        "1": cmd_index,
        "2": cmd_build_dataset,
        "3": cmd_train_ml,
        "4": cmd_train_cnn,
        "5": cmd_evaluate,
        "6": cmd_predict,
        "7": cmd_plots,
        "8": cmd_full,
        "9": cmd_experiment_invert,
    }
    while True:
        try:
            choice = input(MENU).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if choice == "0":
            print("Вихід.")
            break
        fn = actions.get(choice)
        if fn is None:
            print("Невідомий вибір.")
            continue
        try:
            fn()
        except Exception as e:
            print(f"[!] Помилка: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
