# generate_accuracy_across_k.py

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

current_file = Path(__file__).resolve()

project_root = current_file.parents[1]

subjects = [f"sub-{i:02d}" for i in range(1, 11)]

def load_results_across_k(k_features_range, base_path):

    results_across_k = {subject: [] for subject in subjects}

    for k_features in k_features_range:
        for subject in subjects:

            file_path = base_path / f"k_features_{k_features}" / f"{subject}_svm_results_.pkl"

            if not file_path.exists():
                print(f"Arquivo não encontrado: {file_path}")
                continue

            with open(file_path, "rb") as f:
                results = pickle.load(f)
           
            mean_accuracy = results[subject]["mean_accuracy"]

            results_across_k[subject].append(mean_accuracy)

    return results_across_k


def plot_accuracy_across_k(results_across_k, k_features, output_path):

    plt.figure(figsize=(12, 6))

    for subject in subjects:
        accuracies = results_across_k[subject]

        if len(accuracies) != len(k_features):
            continue

        plt.plot(
            k_features,
            np.array(accuracies),
            marker="o",
            label=subject
        )

    plt.title(f"Mean SVM Accuracy across k_features")
    plt.xlabel("k_features")
    plt.ylabel("Mean accuracy")
    #plt.xticks(k_features)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Subject", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    save_path = output_path / f"mean_accuracy_across_k.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Imagem salva em: {save_path}")


def plot_top_n_accuracies(results_across_k, k_features, output_path, top_n = 3):

    top_results = {}

    for subject in subjects:
        mean_accuracies = results_across_k[subject]

        if len(mean_accuracies) != len(k_features):
            continue

        mean_accuracies = np.array(mean_accuracies)

        top_indices = np.argsort(mean_accuracies)[-top_n:][::-1]

        top_results[subject] = [
            {
                "rank": rank + 1,
                "k_feature": int(k_features[index]),
                "mean_accuracy": float(mean_accuracies[index])
            }
            for rank, index in enumerate(top_indices)
        ]

    subjects_names = list(top_results.keys())
    x = np.arange(len(subjects_names))

    plt.figure(figsize=(14, 7))
    width = 0.8 / top_n  # Largura de cada barra individual

    # Iteramos sobre cada nível do rank (0 para Top 1, 1 para Top 2, etc.)
    for rank in range(top_n):
        accuracies = []
        k_values = []

        # Buscamos o valor desse rank específico para todos os sujeitos
        for subject in subjects_names:
            # Garante que o sujeito tem dados para esse rank (caso tenha menos dados que top_n)
            if rank < len(top_results[subject]):
                top_item = top_results[subject][rank]
                accuracies.append(top_item["mean_accuracy"])
                k_values.append(top_item["k_feature"])
            else:
                accuracies.append(0)
                k_values.append(0)

        # Plota as barras do respectivo Rank deslocadas no eixo X
        bars = plt.bar(
            x + (rank * width),
            accuracies,
            width=width,
            label=f"Top {rank + 1}"
        )

        # Adiciona o texto descritivo "k=Valor" acima de cada barra
        for bar, k_value in zip(bars, k_values):
            if k_value > 0:  # Só adiciona o texto se houver dado válido
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,  # Pequeno espaçamento acima da barra
                    f"k={k_value}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90
                )

    # Centraliza os nomes dos sujeitos embaixo do grupo de barras correspondente
    plt.xticks(
        x + (width * (top_n - 1) / 2),
        subjects_names,
        rotation=45
    )

    plt.xlabel("Subject")
    plt.ylabel("Mean accuracy")
    plt.title(f"Top {top_n} accuracies by subject")

    plt.legend(title="Rank")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    save_path = output_path / f"top_{top_n}_accuracies_with_k.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Imagem salva em: {save_path}")   


def generate_accuracy_accross_k_plot(
        k_features_range =[10,20,30,40],
        top_n = 3
):
    
    base_path = (
        project_root
        / "processed_data"
        / "stage_03_svm"
    )

    output_path = (
        project_root
        / "results"
        / "generate_accuracy_across_k"
    )

    k_features_range = list(k_features_range)

    os.makedirs(output_path, exist_ok=True)

    results_across_k = load_results_across_k(k_features_range, base_path)
    plot_accuracy_across_k(results_across_k, k_features_range, output_path)       
    plot_top_n_accuracies(results_across_k, k_features_range, output_path, top_n = top_n)


if __name__ == "__main__":
    generate_accuracy_accross_k_plot()
