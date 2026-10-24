from __future__ import annotations

import os
import sys
import time
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _env_utf8() -> dict:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_branch_inprocess(branch: str, args: argparse.Namespace) -> dict:
    if branch == "image":
        from src.branch_image import run_image_branch
        return run_image_branch(
            do_ablation=not args.no_ablation,
            do_hog=not args.no_hog,
            do_transfer=not args.no_transfer,
            do_cnn=not args.no_cnn,
            backbone=args.backbone,
            cnn_epochs=args.cnn_epochs,
            n_splits=args.splits,
        )
    if branch == "features":
        from src.branch_features import run_feature_branch
        return run_feature_branch(n_splits=args.splits)
    raise ValueError(f"Невідома гілка: {branch}")


def _child_cmd(branch: str, args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "-u", os.path.abspath(__file__),
           "--branch", branch, "--splits", str(args.splits)]
    if branch == "image":
        cmd += ["--backbone", args.backbone,
                "--cnn-epochs", str(args.cnn_epochs)]
        for flag, on in (("--no-ablation", args.no_ablation),
                         ("--no-hog", args.no_hog),
                         ("--no-transfer", args.no_transfer),
                         ("--no-cnn", args.no_cnn)):
            if on:
                cmd.append(flag)
    return cmd


def run_both_parallel(args: argparse.Namespace) -> int:
    print("=" * 62)
    print("  Паралельний запуск: image-гілка + features-гілка")
    print("=" * 62)

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "logs")
    os.makedirs(log_dir, exist_ok=True)

    procs = {}
    for branch in ("image", "features"):
        log_path = os.path.join(log_dir, f"branch_{branch}.log")
        log = open(log_path, "w", encoding="utf-8")
        p = subprocess.Popen(_child_cmd(branch, args), env=_env_utf8(),
                             stdout=log, stderr=subprocess.STDOUT)
        procs[branch] = (p, log, log_path)
        print(f"  [{branch:8s}] pid={p.pid}  лог: "
              f"{os.path.relpath(log_path)}")

    print("\n  Чекаю завершення обох гілок...\n")
    t0 = time.time()
    codes = {}
    for branch, (p, log, log_path) in procs.items():
        code = p.wait()
        log.close()
        codes[branch] = code
        status = "OK" if code == 0 else f"ПОМИЛКА (код {code})"
        print(f"  [{branch:8s}] {status}  "
              f"({time.time() - t0:.0f} с від старту)")
        if code != 0:
            print(f"             дивіться {os.path.relpath(log_path)}")

    print()
    print_summary()
    return 0 if all(c == 0 for c in codes.values()) else 1


def print_summary() -> None:
    import pandas as pd
    from src import config as cfg
    from src.image_methods import IMAGE_RESULTS_CSV, IMAGE_ABLATION_CSV
    from src.branch_features import FEATURES_RESULTS_CSV

    def _show(title: str, path: str, cols: list[str]) -> pd.DataFrame | None:
        if not os.path.exists(path):
            print(f"[--] {title}: ще не рахувалось ({cfg.to_rel(path)})")
            return None
        df = pd.read_csv(path)
        have = [c for c in cols if c in df.columns]
        print(f"\n### {title}  ({cfg.to_rel(path)})")
        print(df[have].to_string(index=False))
        return df

    print("=" * 62)
    print("  ЗВЕДЕНІ РЕЗУЛЬТАТИ")
    print("=" * 62)

    _show("Ablation представлення входу (Група 0)", IMAGE_ABLATION_CSV,
          ["spec", "resize_mode", "channels", "size",
           "pulse_f1_oof", "image_f1_oof", "delta_vs_baseline"])
    img = _show("Image-гілка", IMAGE_RESULTS_CSV,
                ["method", "spec", "pulse_f1_oof", "image_f1_oof",
                 "image_f1_fold_std", "train_sec"])
    feat = _show("Features-гілка", FEATURES_RESULTS_CSV,
                 ["method", "level", "f1_macro", "f1_macro_std", "accuracy"])

    rows = []
    if img is not None and not img.empty:
        best = img.sort_values("image_f1_oof", ascending=False).iloc[0]
        rows.append({"branch": "image", "method": best["method"],
                     "input": best["spec"],
                     "image_f1": round(float(best["image_f1_oof"]), 4)})
    if feat is not None and not feat.empty:
        sub = feat[feat["level"].astype(str).str.startswith("image")]
        if not sub.empty:
            best = sub.sort_values("f1_macro", ascending=False).iloc[0]
            rows.append({"branch": "features", "method": best["method"],
                         "input": "35 числових ознак",
                         "image_f1": round(float(best["f1_macro"]), 4)})
    if rows:
        out = pd.DataFrame(rows).sort_values("image_f1", ascending=False)
        print("\n### Найкраще з кожної гілки (image-level)")
        print(out.to_string(index=False))
        path = os.path.join(cfg.RESULTS_DIR, "branch_comparison.csv")
        out.to_csv(path, index=False)
        print(f"\n[OK] {cfg.to_rel(path)}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Раннер гілок дослідження (image / features).")
    ap.add_argument("--branch", choices=["image", "features", "both"],
                    default="both", help="яку гілку запустити")
    ap.add_argument("--splits", type=int, default=5,
                    help="кількість фолдів GroupKFold (default: 5)")
    ap.add_argument("--summary", action="store_true",
                    help="лише вивести зведену таблицю, нічого не рахувати")
    ap.add_argument("--compare-all", action="store_true",
                    help="єдина таблиця всіх моделей обох гілок "
                         "(results/all_models_comparison.csv); 2D-CNN "
                         "береться з кешу, якщо не задано --with-cnn")
    ap.add_argument("--with-cnn", action="store_true",
                    help="для --compare-all: перерахувати 2D-CNN (~25 хв)")
    ap.add_argument("--save-image-models", action="store_true",
                    help="навчити картинкові моделі на всіх даних і "
                         "зберегти в models/ (для /predict)")
    ap.add_argument("--backbone", default="MobileNetV2",
                    choices=["MobileNetV2", "EfficientNetB0", "ResNet50"])
    ap.add_argument("--cnn-epochs", type=int, default=60)
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--no-hog", action="store_true")
    ap.add_argument("--no-transfer", action="store_true")
    ap.add_argument("--no-cnn", action="store_true")
    args = ap.parse_args()

    if args.summary:
        print_summary()
        return 0

    if args.save_image_models:
        from src.image_model_store import save_all
        t0 = time.time()
        saved = save_all()
        print(f"\n[OK] Збережено моделей: {len(saved)} "
              f"за {time.time() - t0:.0f} с")
        return 0

    if args.compare_all:
        from src.compare_all import compare_all
        compare_all(n_splits=args.splits, run_cnn=args.with_cnn)
        return 0

    if args.branch == "both":
        return run_both_parallel(args)

    t0 = time.time()
    run_branch_inprocess(args.branch, args)
    print(f"\n[OK] Гілка '{args.branch}' завершена за {time.time() - t0:.0f} с")
    return 0


if __name__ == "__main__":
    sys.exit(main())
