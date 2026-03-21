"""
ecg_dataset.py — ECG Benchmark para MANIFOLD V5 (SIGNAL FITTING)

Refactorizado para:
- Leer dataset transpuesto: cada columna (1-42) es una muestra de señal.
- Eliminar lógica de clasificación binaria.
- Retornar (signal, signal) o similar para entrenamiento auto-supervisado/fitting.
"""
import torch
import csv
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional
import os


class ECGSignalDataset(Dataset):
    """
    Dataset ECG que trata cada columna del CSV como una muestra independiente.
    Diseñado para Geodesic Signal Fitting (reconstrucción de dinámica).
    """

    def __init__(
        self,
        csv_path: str,
        seq_len: Optional[int] = None,
        max_samples: Optional[int] = None,
        normalize: bool = True,
        apply_filter: bool = True,
    ):
        self.csv_path = csv_path
        
        # 1. Leer CSV de forma eficiente (Columnas como muestras)
        # Saltamos la cabecera (skip_header=1) y la primera columna (ID)
        data_full = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
        # La primera columna suele ser el ID o tiempo, la saltamos: [Timesteps, ID+Samples] -> [Timesteps, Samples]
        data_matrix = data_full[:, 1:].T # [Samples, Timesteps]
        
        if max_samples:
            data_matrix = data_matrix[:max_samples]
            
        print(f"[ECG Dataset] Matrix cargada: {data_matrix.shape} (Samples x Timesteps)")

        # 2. Preprocesamiento (Filtro)
        if apply_filter:
            data_matrix = np.array([self._butterworth_bandpass(s, fs=500) for s in data_matrix])

        # 3. Normalización Global
        if normalize:
            self.mean = data_matrix.mean()
            self.std = data_matrix.std() + 1e-8
            data_matrix = (data_matrix - self.mean) / self.global_std if hasattr(self, 'global_std') else (data_matrix - self.mean) / self.std
            print(f"[ECG Dataset] Normalizacion: mean={self.mean:.4f}, std={self.std:.4f}")

        # 4. Preparar Tensores
        self.seq_len = seq_len if seq_len else data_matrix.shape[1]
        self.signals = []
        
        for i in range(data_matrix.shape[0]):
            sig = data_matrix[i, :self.seq_len].astype(np.float32)
            self.signals.append(torch.from_numpy(sig).unsqueeze(-1)) # [L, 1]

        print(f"[ECG Dataset] {len(self.signals)} muestras de longitud {self.seq_len}")

    def _butterworth_bandpass(self, signal: np.ndarray, fs: float = 500) -> np.ndarray:
        try:
            from scipy.signal import butter, filtfilt
            low = 0.5 / (fs / 2)
            high = 40 / (fs / 2)
            if high >= 1.0: high = 0.95
            b, a = butter(4, [low, high], btype='band')
            return filtfilt(b, a, signal)
        except ImportError:
            return signal

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Para Signal Fitting: input es la señal, target es la señal misma (reconstrucción)
        sig = self.signals[idx]
        return sig, sig


# Mantener alias para compatibilidad si es necesario
ECGTimeseriesDataset = ECGSignalDataset

if __name__ == "__main__":
    csv_path = r"D:\ASAS\manifold_mini\manifold_working\tests\benchmarks\convergence\healt\datasets\ECG Timeseries-20260303T021501Z-1-001\ECG Timeseries\ecg_timeseries.csv"
    if os.path.exists(csv_path):
        ds = ECGSignalDataset(csv_path, max_samples=5)
        sig, target = ds[0]
        print(f"Signal shape: {sig.shape}")
    else:
        print("CSV no encontrado")
