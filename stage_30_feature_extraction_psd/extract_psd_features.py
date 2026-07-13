import os
import numpy as np
from scipy.signal import periodogram
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parents[1]

input_path = project_root / "processed_data" / "stage_20_band_filtering_and_epoch"

output_path = project_root / "processed_data" / current_file.parents[0].name

os.makedirs(output_path, exist_ok=True)

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
sessions = [f"ses-{i:02d}" for i in range(1, 4)]
fs = 256

def extract_and_save_features():
    for subject in subjects:
        for session in sessions:
            
            data_path = input_path / f"{subject}_{session}_inner_bands.npy"
            labels_path = input_path / f"{subject}_{session}_inner_bands_labels.npy"

            if not data_path.exists() or not labels_path.exists():
                continue
            
            print(f"Processando PSD: {subject} {session}...")
            
            X_bands = np.load(data_path)
            labels = np.load(labels_path)
            
            freqs, Pxx = periodogram(X_bands, fs=fs, axis=-1)

            alpha_mask = (freqs >= 8) & (freqs <= 12)
            beta_mask = (freqs >= 12) & (freqs <= 30)
            
            # Isolar a matriz de cada banda primeiro
            Pxx_alpha = Pxx[:, 0, :, :]
            Pxx_beta = Pxx[:, 1, :, :]
            
            #Aplicar a máscara e somar no eixo da frequência
            alpha_power = Pxx_alpha[:, :, alpha_mask].sum(axis=-1)
            beta_power = Pxx_beta[:, :, beta_mask].sum(axis=-1)
            
            # Juntar as features de Alpha e Beta num único vetor
            X_features = np.concatenate((alpha_power, beta_power), axis=1)

            save_data_path = output_path / f"{subject}_{session}_features_psd.npy"
            save_labels_path = output_path / f"{subject}_{session}_labels.npy"
            
            np.save(save_data_path, X_features)
            np.save(save_labels_path, labels)
            
            print(f" -> Extraído: {X_features.shape[0]} épocas | {X_features.shape[1]} features (Alpha + Beta).")

    print(f"\nFinalizado {current_file.parents[0].name}.")

if __name__ == "__main__":
    extract_and_save_features()