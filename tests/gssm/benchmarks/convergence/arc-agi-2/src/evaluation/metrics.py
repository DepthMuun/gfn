"""
ARC-AGI-2 Evaluation Metrics
Implementación VERIFICABLE de métricas para ARC-AGI-2.

ESTA ES LA IMPLEMENTACIÓN OFICIAL - NO MODIFICAR sin revisión.
La métrica principal es strict match: TODOS los pixels deben coincidir exactamente.
Las métricas se evalúan sobre la región original del grid (sin padding).
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
import json


def crop_to_original(grid: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
    """
    Crop a padded grid to its original size.

    Args:
        grid: Padded grid (max_size, max_size) or (H, W)
        original_size: (H, W) original size before padding

    Returns:
        Cropped grid of shape original_size
    """
    h, w = original_size
    if h <= grid.shape[0] and w <= grid.shape[1]:
        return grid[:h, :w]
    return grid


class ARCMetrics:
    """
    Métricas para evaluación ARC-AGI-2.

    IMPORTANTE: La métrica oficial de ARC-AGI-2 es strict match.
    No hay partial credit. Size, colores y posiciones deben coincidir perfectamente.
    Todas las métricas se calculan sobre la región original (sin padding).
    """

    @staticmethod
    def strict_match(prediction: np.ndarray, ground_truth: np.ndarray) -> bool:
        """
        Verifica si la predicción es EXACTAMENTE igual al ground truth.

        Args:
            prediction: Grid predicho (H, W)
            ground_truth: Ground truth grid (H, W)

        Returns:
            True si son idénticos, False en caso contrario
        """
        if prediction.shape != ground_truth.shape:
            return False
        return np.array_equal(prediction, ground_truth)

    @staticmethod
    def pixel_accuracy(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Porcentaje de pixels correctos.
        NOTA: Esta no es la métrica oficial, solo informativa.

        Args:
            prediction: Grid predicho
            ground_truth: Ground truth

        Returns:
            Accuracy entre 0.0 y 1.0
        """
        if prediction.shape != ground_truth.shape:
            return 0.0
        correct = np.sum(prediction == ground_truth)
        total = ground_truth.size
        return correct / total

    @staticmethod
    def size_accuracy(predicted_size: Tuple[int, int], true_size: Tuple[int, int]) -> bool:
        """
        Verifica si el tamaño predicho es correcto.

        Args:
            predicted_size: (H, W) predicho
            true_size: (H, W) verdadero

        Returns:
            True si los tamaños coinciden
        """
        return predicted_size == true_size

    @staticmethod
    def evaluate_task(
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        pred_size: Optional[Tuple[int, int]] = None,
        true_size: Optional[Tuple[int, int]] = None
    ) -> Dict:
        """
        Evalúa una task completa con todas las métricas.
        Evalúa sobre la región original (sin padding) cuando se proporcionan tamaños.

        Args:
            prediction: Grid predicho (puede ser padded a max_size x max_size)
            ground_truth: Ground truth (puede ser padded)
            pred_size: Tamaño original predicho (H, W). Si es None, usa prediction.shape
            true_size: Tamaño original verdadero (H, W). Si es None, usa ground_truth.shape

        Returns:
            Dict con métricas calculadas sobre la región original
        """
        # Usar tamaños de los arrays si no se proporcionan
        if pred_size is None:
            pred_size = (prediction.shape[0], prediction.shape[1])
        if true_size is None:
            true_size = (ground_truth.shape[0], ground_truth.shape[1])

        # Crop a tamaño original para evaluación sin padding
        pred_cropped = crop_to_original(prediction, pred_size)
        gt_cropped = crop_to_original(ground_truth, true_size)

        # Size accuracy: los tamaños originales deben coincidir
        size_correct = ARCMetrics.size_accuracy(pred_size, true_size)

        # Strict match sobre región original (no padding)
        strict_correct = False
        if size_correct:
            strict_correct = ARCMetrics.strict_match(pred_cropped, gt_cropped)

        # Pixel accuracy sobre región original
        pixel_acc = ARCMetrics.pixel_accuracy(pred_cropped, gt_cropped)

        return {
            'strict_match': strict_correct,
            'pixel_accuracy': pixel_acc,
            'size_correct': size_correct,
            'correct_pixels': int(np.sum(pred_cropped == gt_cropped)),
            'total_pixels': int(gt_cropped.size)
        }

    @staticmethod
    def aggregate_metrics(task_results: List[Dict]) -> Dict:
        """
        Agrega métricas de múltiples tasks.

        Args:
            task_results: Lista de resultados por task

        Returns:
            Dict con métricas agregadas
        """
        if not task_results:
            return {
                'task_accuracy': 0.0,
                'mean_pixel_accuracy': 0.0,
                'size_accuracy': 0.0,
                'num_tasks': 0,
                'tasks_correct': 0
            }

        num_tasks = len(task_results)

        tasks_correct = sum(1 for r in task_results if r['strict_match'])
        task_accuracy = tasks_correct / num_tasks

        mean_pixel_acc = float(np.mean([r['pixel_accuracy'] for r in task_results]))

        sizes_correct = sum(1 for r in task_results if r['size_correct'])
        size_accuracy = sizes_correct / num_tasks

        return {
            'task_accuracy': task_accuracy,
            'tasks_correct': tasks_correct,
            'mean_pixel_accuracy': mean_pixel_acc,
            'size_accuracy': size_accuracy,
            'num_tasks': num_tasks
        }


def save_predictions(
    predictions: Dict[str, np.ndarray],
    output_path: str,
    original_sizes: Optional[Dict[str, Tuple[int, int]]] = None
):
    """
    Guarda predicciones en formato JSON para evaluación.
    Las predicciones se guardan en su tamaño original (sin padding).

    Args:
        predictions: Dict {task_id: grid} (puede ser padded)
        output_path: Ruta de salida
        original_sizes: Dict {task_id: (H, W)} tamaños originales
    """
    output = {}
    for task_id, grid in predictions.items():
        grid_list = grid.tolist()

        # Crop a tamaño original si está disponible
        if original_sizes and task_id in original_sizes:
            h, w = original_sizes[task_id]
            grid_list = [row[:w] for row in grid_list[:h]]

        output[task_id] = {
            'prediction': grid_list,
            'shape': [len(grid_list), len(grid_list[0]) if grid_list else 0]
        }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


def load_predictions(predictions_path: str) -> Dict[str, np.ndarray]:
    """Carga predicciones desde JSON."""
    with open(predictions_path, 'r') as f:
        data = json.load(f)

    predictions = {}
    for task_id, task_data in data.items():
        predictions[task_id] = np.array(task_data['prediction'])

    return predictions


def verify_metrics_implementation():
    """
    Verifica que las métricas estén implementadas correctamente.
    Test cases con resultados conocidos.
    """
    print("Verifying ARC-AGI-2 metrics implementation...")

    # Test 1: Perfect match
    pred1 = np.array([[1, 2], [3, 4]])
    gt1 = np.array([[1, 2], [3, 4]])
    assert ARCMetrics.strict_match(pred1, gt1) == True, "Test 1 failed: perfect match"

    # Test 2: One pixel off
    pred2 = np.array([[1, 2], [3, 5]])
    gt2 = np.array([[1, 2], [3, 4]])
    assert ARCMetrics.strict_match(pred2, gt2) == False, "Test 2 failed: one pixel off"

    # Test 3: Different size
    pred3 = np.array([[1, 2, 3]])
    gt3 = np.array([[1, 2]])
    assert ARCMetrics.strict_match(pred3, gt3) == False, "Test 3 failed: different size"

    # Test 4: Pixel accuracy
    pred4 = np.array([[1, 2], [3, 5]])
    gt4 = np.array([[1, 2], [3, 4]])
    acc = ARCMetrics.pixel_accuracy(pred4, gt4)
    assert abs(acc - 0.75) < 0.01, f"Test 4 failed: pixel accuracy should be 0.75, got {acc}"

    # Test 5: Padded grids evaluated on original region
    # A 2x2 grid padded to 30x30 - padding zeros should NOT inflate accuracy
    pred5 = np.zeros((30, 30), dtype=np.int64)
    pred5[:2, :2] = [[1, 2], [3, 4]]
    gt5 = np.zeros((30, 30), dtype=np.int64)
    gt5[:2, :2] = [[1, 2], [3, 4]]
    metrics5 = ARCMetrics.evaluate_task(pred5, gt5, pred_size=(2, 2), true_size=(2, 2))
    assert metrics5['strict_match'] == True, "Test 5a failed: padded perfect match"
    assert metrics5['pixel_accuracy'] == 1.0, f"Test 5b failed: pixel acc should be 1.0, got {metrics5['pixel_accuracy']}"
    assert metrics5['total_pixels'] == 4, f"Test 5c failed: total_pixels should be 4, got {metrics5['total_pixels']}"

    # Test 6: Padded grid with wrong content in original region
    pred6 = np.zeros((30, 30), dtype=np.int64)
    pred6[:2, :2] = [[1, 2], [3, 99]]
    gt6 = np.zeros((30, 30), dtype=np.int64)
    gt6[:2, :2] = [[1, 2], [3, 4]]
    metrics6 = ARCMetrics.evaluate_task(pred6, gt6, pred_size=(2, 2), true_size=(2, 2))
    assert metrics6['strict_match'] == False, "Test 6a failed: padded mismatch should be False"
    assert abs(metrics6['pixel_accuracy'] - 0.75) < 0.01, f"Test 6b failed: pixel acc should be 0.75, got {metrics6['pixel_accuracy']}"
    assert metrics6['total_pixels'] == 4, f"Test 6c failed: total_pixels should be 4, got {metrics6['total_pixels']}"

    print("All metric tests passed!")
    print("Implementation verified: STRICT MATCH is the official metric.")
    print("Evaluation uses original grid region (no padding inflation).")


if __name__ == "__main__":
    verify_metrics_implementation()
