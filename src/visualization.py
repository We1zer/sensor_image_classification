import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from . import config as cfg
from .image_preprocessing import read_image, preprocess_pipeline
from .signal_extraction import mask_to_clean_signal
from .pulse_segmentation import find_pulse_boundaries


def _save(fig, name: str) -> str:
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    out = os.path.join(cfg.RESULTS_DIR, name)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pipeline_example(image_path: str,
                          out_name: str = "signal_extraction_example.png",
                          invert: bool = False) -> str:
    img = read_image(image_path)
    prep = preprocess_pipeline(img, invert=invert)
    cropped = prep["cropped"]
    mask = prep["mask"]
    x, y, _ = mask_to_clean_signal(mask, do_smooth=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].imshow(img[:, :, ::-1])
    axes[0, 0].set_title("Оригінал")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cropped[:, :, ::-1])
    axes[0, 1].set_title("Обрізана область графіка")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(mask, cmap="gray")
    axes[1, 0].set_title("Маска сигналу (HSV / threshold)")
    axes[1, 0].axis("off")

    axes[1, 1].plot(x, y, lw=1.2)
    axes[1, 1].set_title("Числовий сигнал")
    axes[1, 1].set_xlabel("x (пікселі)")
    axes[1, 1].set_ylabel("амплітуда")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Приклад обробки: {os.path.basename(image_path)}", fontsize=13)
    return _save(fig, out_name)


def plot_pulse_segmentation_example(image_path: str,
                                    out_name: str = "examples_of_detected_pulses.png",
                                    invert: bool = False) -> str:
    img = read_image(image_path)
    prep = preprocess_pipeline(img, invert=invert)
    x, y, _ = mask_to_clean_signal(prep["mask"])
    bounds = find_pulse_boundaries(y)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, y, lw=1.2, color="#1f77b4", label="сигнал")
    for (a, b) in bounds:
        ax.axvspan(a, b, color="orange", alpha=0.25)
    ax.set_title(f"Сегментація на імпульси: {os.path.basename(image_path)}")
    ax.set_xlabel("x (пікселі)")
    ax.set_ylabel("амплітуда")
    ax.grid(True, alpha=0.3)
    return _save(fig, out_name)


def plot_class_distribution(df: pd.DataFrame,
                            out_name: str = "class_distribution.png") -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df["class"].value_counts().reindex(cfg.CLASS_NAMES, fill_value=0)
    sns.barplot(x=counts.index, y=counts.values, ax=ax,
                hue=counts.index, palette="viridis", legend=False)
    ax.set_title("Кількість імпульсів за класами")
    ax.set_xlabel("клас")
    ax.set_ylabel("кількість імпульсів")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom")
    return _save(fig, out_name)


def plot_feature_distributions(df: pd.DataFrame,
                               features: list[str] | None = None,
                               out_name: str = "feature_distributions.png") -> str:
    if features is None:
        features = ["amplitude", "area_under_curve",
                    "rise_time", "fall_time",
                    "energy", "std_val"]
    n = len(features)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, f in zip(axes, features):
        if f not in df.columns:
            ax.set_visible(False)
            continue
        sns.boxplot(x="class", y=f, data=df, ax=ax,
                    order=cfg.CLASS_NAMES, hue="class", palette="Set2",
                    legend=False)
        ax.set_title(f)
        ax.set_xlabel("")
    for ax in axes[len(features):]:
        ax.set_visible(False)
    fig.suptitle("Розподіл ознак за класами", fontsize=13)
    fig.tight_layout()
    return _save(fig, out_name)


def plot_feature_correlation(df: pd.DataFrame,
                             out_name: str = "feature_correlation_matrix.png") -> str:
    from .feature_extraction import FEATURE_NAMES
    cols = [c for c in FEATURE_NAMES if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Матриця кореляції ознак")
    return _save(fig, out_name)


def plot_confusion(cm: np.ndarray, class_names: list[str],
                   title: str = "Confusion matrix",
                   out_name: str = "confusion_matrix.png") -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Прогноз")
    ax.set_ylabel("Істинний клас")
    ax.set_title(title)
    return _save(fig, out_name)


def plot_model_comparison(results_df: pd.DataFrame,
                          out_name: str = "model_comparison.png") -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = results_df.melt(id_vars=["model"],
                              value_vars=["accuracy", "precision_macro",
                                          "recall_macro", "f1_macro"],
                              var_name="metric", value_name="value")
    sns.barplot(data=plot_df, x="model", y="value", hue="metric", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Порівняння моделей")
    ax.set_ylabel("значення метрики")
    ax.set_xlabel("")
    plt.xticks(rotation=20)
    return _save(fig, out_name)


def plot_concentration_analysis(df: pd.DataFrame,
                                feature: str = "amplitude",
                                out_name: str = "concentration_analysis.png") -> str:
    classes = [c for c in cfg.CLASS_NAMES if c != "RH"]
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4),
                             sharey=True)
    if len(classes) == 1:
        axes = [axes]
    for ax, cls in zip(axes, classes):
        sub = df[df["class"] == cls]
        if sub.empty:
            ax.set_title(f"{cls}: немає даних")
            continue
        agg = sub.groupby("file_number")[feature].mean().reset_index()
        ax.plot(agg["file_number"], agg[feature], "-o")
        ax.set_title(f"{cls}: {feature} vs file_number")
        ax.set_xlabel("file_number (≈ концентрація)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(feature)
    fig.suptitle("Залежність ознаки від номера файлу (концентрації)")
    fig.tight_layout()
    return _save(fig, out_name)


def generate_all_plots(features_df: pd.DataFrame,
                       results_dict: dict | None = None,
                       example_image: str | None = None) -> list[str]:
    saved = []
    if not features_df.empty:
        saved.append(plot_class_distribution(features_df))
        saved.append(plot_feature_distributions(features_df))
        saved.append(plot_feature_correlation(features_df))
        saved.append(plot_concentration_analysis(features_df,
                                                 feature="amplitude",
                                                 out_name="concentration_analysis_amplitude.png"))
        saved.append(plot_concentration_analysis(features_df,
                                                 feature="area_under_curve",
                                                 out_name="concentration_analysis_area.png"))

    if results_dict is not None and "results" in results_dict:
        saved.append(plot_model_comparison(results_dict["results"]))
        cm = results_dict.get("confusion")
        cn = results_dict.get("class_names")
        if cm is not None and cn is not None:
            saved.append(plot_confusion(cm, cn,
                                        title=f"Confusion: {results_dict['best_name']}",
                                        out_name=f"confusion_matrix_{results_dict['best_name']}.png"))

    if example_image and os.path.exists(example_image):
        saved.append(plot_pipeline_example(example_image))
        saved.append(plot_pulse_segmentation_example(example_image))

    return saved
