import os
import numpy as np
from pathlib import Path

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

import pickle

current_file = Path(__file__).resolve()
project_root = current_file.parents[1]

base_path = project_root / "processed_data" / "stage_30_feature_extraction_psd"
output_path = project_root / "processed_data" / current_file.parents[0].name

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
sessions = [f"ses-{i:02d}" for i in range(1, 4)]

def run_psd_feature_selection_SVM(
    k_features = 3,
    feature_selector = "f_classif",
    output_path = output_path,
    ):

    os.makedirs(output_path, exist_ok=True)

    for subject in subjects:

        results = {}
        graph_sessions = []
        label_sessions = []

        for session in sessions:

            """___Abrindo o aquivo a ser processado___"""
            file_path = os.path.join(
                base_path,
                f"{subject}_{session}_features_psd.npy"
            )

            labels_path = os.path.join(
                base_path,
                f"{subject}_{session}_labels.npy"
            )

            if not os.path.exists(file_path):
                print(f"Arquivo não encontrado: {file_path}")
                continue

            if not os.path.exists(labels_path):
                print(f"Rótulos não encontrado: {labels_path}")
                continue

            graph_measures = np.load(file_path)
            labels = np.load(labels_path)

            graph_sessions.append(graph_measures)
            label_sessions.append(labels)

        # Unindo as 3 sessões do sujeito
        if len(graph_sessions) != len(sessions):
            print(f"Pulando {subject}: sessões incompletas")
            continue

        subject_graph_measures = np.concatenate(graph_sessions, axis=0)
        labels = np.concatenate(label_sessions, axis=0)

        if subject_graph_measures.shape[0] != labels.shape[0]:
            raise ValueError(
                f"{subject}: número de épocas diferente do número de rótulos"
            )
        
        print(f"\n{subject} \nShape dos dados: {subject_graph_measures.shape} \nShape dos rótulos: {labels.shape}")

        X = subject_graph_measures
        labels_current = labels.copy()

        steps = [
            ("scaler", StandardScaler())
        ]

        selector = None
        if feature_selector == "f_classif":
            selector = SelectKBest(score_func=f_classif, k=k_features)
        elif feature_selector == "mutual_info_classif":
            selector = SelectKBest(score_func=mutual_info_classif, k=k_features)

        if selector is not None:
            steps.append(
                ("selector", selector)
            )            

        steps.append(
            ("svm", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight=None,
                tol=1e-3,
                max_iter=-1
            ))
        )

        model = Pipeline(steps)

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        scores = []
        confusion_matrices = []

        selected_features_all_folds = []

        for train_idx, test_idx in cv.split(X, labels_current):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = labels_current[train_idx], labels_current[test_idx]

            model.fit(X_train, y_train)

            # Salvando as features utilizadas
            selector_step = model.named_steps.get("selector")
            feature_info = []

            if selector_step is not None:
                selected_features = np.where(
                    selector_step.get_support()
                )[0]

                for feature_idx in selected_features:
                    band_idx = 0 if feature_idx < 128 else 1
                    channel_idx = feature_idx % 128
                    band_name = "Alpha" if band_idx == 0 else "Beta"

                    feature_info.append({
                        "feature_idx": int(feature_idx),
                        "band_index": band_idx,
                        "band_name": band_name,
                        "channel_index": int(channel_idx),
                        "channel_name": f"CH_{channel_idx}"
                    })

            selected_features_all_folds.append(feature_info)

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            scores.append(acc)

            cm = confusion_matrix(
                y_test,
                y_pred,
                labels=np.unique(labels_current),
                normalize="true"
            )

            confusion_matrices.append(cm)

        scores = np.array(scores)
        confusion_matrices = np.array(confusion_matrices)

        mean_confusion_matrix = confusion_matrices.mean(axis=0)

        results[subject] = {
            "k_features": k_features,
            "scores": scores,
            "mean_accuracy": scores.mean(),
            "std_accuracy": scores.std(),
            "confusion_matrices": confusion_matrices,
            "mean_confusion_matrix": mean_confusion_matrix,
            "selected_features": selected_features_all_folds
        }

        print(
            subject,
            "scores:", scores,
            "mean:", scores.mean()
        )

        save_path = os.path.join(
            output_path,
            f"{subject}_psd_svm_results_k_{k_features}.pkl"
        )

        with open(save_path, "wb") as f:
            pickle.dump(results, f)

if __name__ == "__main__":
    run_psd_feature_selection_SVM()