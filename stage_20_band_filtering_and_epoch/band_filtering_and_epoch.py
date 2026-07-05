import os
import mne
import numpy as np
import pickle
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parents[1]

base_path = project_root / "thinking_outloud_dataset" / "derivatives"
output_path = project_root / "processed_data" / current_file.parents[0].name

os.makedirs(output_path, exist_ok=True)

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
sessions = [f"ses-{i:02d}" for i in range(1, 4)]

def obtain_filtered_bands_and_epochs(epochs: np.ndarray, events_path: np.ndarray) -> np.ndarray:

    bands = [
        (8,12),
        (12,30),
    ]

    with open(events_path, "rb") as f:
            events = pickle.load(f)



    mask_inner = events[:, 2] == 1
    epochs_inner = epochs[mask_inner]
    labels = events[mask_inner, 1]

    fs = 256
    start = int(1.5 * fs)   # 384
    end = int(3.5 * fs)     # 896

    band_arrays = []

    for l_freq, h_freq in bands:
        epochs_band = epochs_inner.copy().filter(
            l_freq=l_freq,
            h_freq=h_freq,
            picks="eeg",
            method="fir",
            phase="zero",
            fir_design="firwin",
            verbose=False
        )

        # data_band.shape = (épocas, canais, tempo)
        data_band = epochs_band.get_data()[:, :, start:end] # [ banda1, banda2 ] com cada banda no formato (épocas × canais × tempo)

        band_arrays.append(data_band)  

    X_bands = np.stack(band_arrays, axis=1) #(época × bandas × canais × tempo)
    
    return X_bands, labels

def run_band_filtering_and_epoch(): 

    for subject in subjects:
        for session in sessions:

            """___Abrindo o aquivo a ser processado___"""

            file_path = os.path.join(
                base_path,
                subject,
                session,
                f"{subject}_{session}_eeg-epo.fif"
            )

            events_path = os.path.join(
                base_path,
                subject,
                session,
                f"{subject}_{session}_events.dat"
            )

            if not os.path.exists(file_path):
                print(f"Arquivo não encontrado: {file_path}")
                continue

            if not os.path.exists(events_path):
                print(f"Arquivo não encontrado: {events_path}")
                continue

            """___Processando os dados___"""

            print(f"\nProcessando {current_file.parents[0].name}\n{subject} {session}...")
            epochs = mne.read_epochs(file_path, preload=True, verbose=False)

            X_bands, labels = obtain_filtered_bands_and_epochs(epochs, events_path) #(épocas × bandas × canais × tempos)


            #imprimindo os canais
            channel_map = {}
            for idx, name in enumerate(epochs.ch_names):
                channel_map[name] = idx
            print(channel_map)
        
            """___Salvando os dados processados__"""
            
            save_path = os.path.join(
                output_path,
                f"{subject}_{session}_inner_bands.npy"
            )

            labels_path = os.path.join(
                output_path,
                f"{subject}_{session}_inner_bands_labels.npy"
)

            np.save(save_path, X_bands)
            np.save(labels_path, labels)

            print(f"Dados salvos em: {save_path}")
            print(f"Rótulos salvos em: {labels_path}")


            print(f"Shape dos dados: {X_bands.shape}")
            print(f"Shape dos rótulos: {labels.shape}")
            print(f"Labels únicos: {np.unique(labels, return_counts=True)}")


    print(f"\nFinalizado {current_file.parents[0].name}.")


if __name__ == "__main__":

    run_band_filtering_and_epoch()