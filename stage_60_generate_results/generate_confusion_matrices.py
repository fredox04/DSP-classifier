"""Generate per-subject and group-level normalized confusion matrices."""

from __future__ import annotations

import csv
import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
INPUT_DIR = PROJECT_ROOT / "processed_data" / "stage_40_svm"
OUTPUT_DIR = PROJECT_ROOT / "results" / CURRENT_FILE.parent.name
CLASS_NAMES = ["Up", "Down", "Right", "Left"]


def load_subject_results() -> dict[str, dict]:
    """Load the SVM result of each subject from the stage 40 pickle files."""
    files = list(INPUT_DIR.glob("sub-*_psd_svm_results_k_*.pkl"))

    def sort_key(path: Path) -> tuple[int, int]:
        match = re.search(r"sub-(\d+).*_k_(\d+)", path.stem)
        return tuple(map(int, match.groups())) if match else (999, 999)

    results_by_subject = {}
    for file_path in sorted(files, key=sort_key):
        with file_path.open("rb") as file:
            payload = pickle.load(file)
        if not isinstance(payload, dict) or len(payload) != 1:
            raise ValueError(f"Unexpected result structure in {file_path}")
        subject, result = next(iter(payload.items()))
        if subject in results_by_subject:
            raise ValueError(f"More than one result was found for {subject}")
        results_by_subject[subject] = result

    if not results_by_subject:
        raise FileNotFoundError(f"No SVM result files found in {INPUT_DIR}")
    return results_by_subject


def annotate_matrix(ax: plt.Axes, matrix: np.ndarray) -> None:
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] > 0.5 else "black"
            ax.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", color=color, fontsize=8)


def configure_axis(ax: plt.Axes, title: str) -> None:
    ticks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(ticks, CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(ticks, CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)


def generate_confusion_matrices() -> None:
    results = load_subject_results()
    output_dir = OUTPUT_DIR / "confusion_matrices"
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = list(results)
    matrices = np.array([results[s]["mean_confusion_matrix"] for s in subjects], dtype=float)
    group_matrix = matrices.mean(axis=0)

    fig, axes = plt.subplots(2, 5, figsize=(16, 7), constrained_layout=True)
    image = None
    for ax, subject, matrix in zip(axes.ravel(), subjects, matrices):
        recalls = np.diag(matrix)
        title = f"{subject}\nbest={CLASS_NAMES[recalls.argmax()]}, worst={CLASS_NAMES[recalls.argmin()]}"
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        annotate_matrix(ax, matrix)
        configure_axis(ax, title)
    fig.suptitle("Mean row-normalized confusion matrices by participant")
    fig.colorbar(image, ax=axes, shrink=0.8, label="Proportion")
    fig.savefig(output_dir / "confusion_matrices_by_subject.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(group_matrix, vmin=0, vmax=1, cmap="Blues")
    annotate_matrix(ax, group_matrix)
    configure_axis(ax, f"Group mean (balanced accuracy = {np.diag(group_matrix).mean() * 100:.1f}%)")
    fig.colorbar(image, ax=ax, label="Proportion")
    fig.tight_layout()
    fig.savefig(output_dir / "group_mean_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    with (output_dir / "group_mean_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, group_matrix):
            writer.writerow([class_name, *(f"{value:.6f}" for value in row)])

    print(f"Confusion matrices saved to {output_dir}")


if __name__ == "__main__":
    generate_confusion_matrices()


