"""
Test Metrics Verification
Tests para verificar que las métricas ARC-AGI-2 están implementadas correctamente.
Este archivo es CRÍTICO para asegurar que las métricas sean correctas antes de publicar resultados.
"""

import sys
from pathlib import Path

# Add the benchmark src directory to the path
BENCHMARK_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(BENCHMARK_ROOT))

import numpy as np
import json
import tempfile

from src.evaluation.metrics import ARCMetrics, save_predictions, load_predictions


class TestARCMetrics:
    """Test suite para métricas ARC-AGI-2."""
    
    def test_strict_match_perfect(self):
        """Test: Predicción perfecta debe dar True."""
        pred = np.array([[1, 2, 3], [4, 5, 6]])
        gt = np.array([[1, 2, 3], [4, 5, 6]])
        
        result = ARCMetrics.strict_match(pred, gt)
        assert result == True, "Perfect match should return True"
        print("✓ Test passed: perfect match")
    
    def test_strict_match_one_pixel_off(self):
        """Test: Un pixel diferente debe dar False."""
        pred = np.array([[1, 2, 3], [4, 5, 99]])  # 99 en lugar de 6
        gt = np.array([[1, 2, 3], [4, 5, 6]])
        
        result = ARCMetrics.strict_match(pred, gt)
        assert result == False, "One pixel off should return False"
        print("✓ Test passed: one pixel off")
    
    def test_strict_match_different_size(self):
        """Test: Tamaño diferente debe dar False."""
        pred = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2
        gt = np.array([[1, 2], [3, 4]])  # 2x2
        
        result = ARCMetrics.strict_match(pred, gt)
        assert result == False, "Different size should return False"
        print("✓ Test passed: different size")
    
    def test_pixel_accuracy_perfect(self):
        """Test: Pixel accuracy 100% para match perfecto."""
        pred = np.array([[1, 2], [3, 4]])
        gt = np.array([[1, 2], [3, 4]])
        
        acc = ARCMetrics.pixel_accuracy(pred, gt)
        assert abs(acc - 1.0) < 0.01, f"Perfect match should have 100% pixel accuracy, got {acc}"
        print("✓ Test passed: pixel accuracy 100%")
    
    def test_pixel_accuracy_partial(self):
        """Test: Pixel accuracy parcial calculada correctamente."""
        pred = np.array([[1, 2], [3, 99]])  # 3/4 correctos
        gt = np.array([[1, 2], [3, 4]])
        
        acc = ARCMetrics.pixel_accuracy(pred, gt)
        expected = 0.75
        assert abs(acc - expected) < 0.01, f"Expected {expected}, got {acc}"
        print("✓ Test passed: pixel accuracy 75%")
    
    def test_size_accuracy(self):
        """Test: Size accuracy."""
        pred_size = (10, 10)
        true_size = (10, 10)
        
        result = ARCMetrics.size_accuracy(pred_size, true_size)
        assert result == True, "Same size should return True"
        print("✓ Test passed: size accuracy")
    
    def test_evaluate_task_complete(self):
        """Test: Evaluación completa de task."""
        pred = np.array([[1, 2], [3, 4]])
        gt = np.array([[1, 2], [3, 4]])
        
        metrics = ARCMetrics.evaluate_task(pred, gt)
        
        assert 'strict_match' in metrics
        assert 'pixel_accuracy' in metrics
        assert 'size_correct' in metrics
        assert metrics['strict_match'] == True
        assert abs(metrics['pixel_accuracy'] - 1.0) < 0.01
        assert metrics['size_correct'] == True
        print("✓ Test passed: complete task evaluation")
    
    def test_aggregate_metrics(self):
        """Test: Agregación de métricas múltiples."""
        task_results = [
            {'strict_match': True, 'pixel_accuracy': 1.0, 'size_correct': True, 'correct_pixels': 4, 'total_pixels': 4},
            {'strict_match': False, 'pixel_accuracy': 0.5, 'size_correct': True, 'correct_pixels': 2, 'total_pixels': 4},
            {'strict_match': True, 'pixel_accuracy': 1.0, 'size_correct': True, 'correct_pixels': 4, 'total_pixels': 4},
            {'strict_match': False, 'pixel_accuracy': 0.0, 'size_correct': False, 'correct_pixels': 0, 'total_pixels': 4},
        ]
        
        aggregated = ARCMetrics.aggregate_metrics(task_results)
        
        # Task accuracy: 2/4 = 0.5
        assert abs(aggregated['task_accuracy'] - 0.5) < 0.01
        assert aggregated['tasks_correct'] == 2
        assert aggregated['num_tasks'] == 4
        print("✓ Test passed: aggregate metrics")
    
    def test_save_and_load_predictions(self):
        """Test: Guardar y cargar predicciones."""
        predictions = {
            'task_001': np.array([[1, 2], [3, 4]]),
            'task_002': np.array([[5, 6, 7], [8, 9, 10]])
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            save_predictions(predictions, temp_path)
            loaded = load_predictions(temp_path)
            
            assert 'task_001' in loaded
            assert 'task_002' in loaded
            assert np.array_equal(loaded['task_001'], predictions['task_001'])
            assert np.array_equal(loaded['task_002'], predictions['task_002'])
            print("✓ Test passed: save and load predictions")
        finally:
            import os
            os.unlink(temp_path)
    
    def test_arc_official_metric(self):
        """
        Test: Verificar que strict_match es la métrica oficial de ARC-AGI-2.
        Según ARC Prize, la métrica oficial es exact match (100% correcto o 0%).
        """
        # Caso 1: Predicción correcta
        pred_correct = np.array([[0, 1, 2], [3, 4, 5]])
        gt = np.array([[0, 1, 2], [3, 4, 5]])
        
        metrics = ARCMetrics.evaluate_task(pred_correct, gt)
        
        # La task debe contar como "correcta"
        assert metrics['strict_match'] == True, "ARC official metric: perfect prediction must count as correct"
        
        # Caso 2: Un solo pixel incorrecto
        pred_incorrect = np.array([[0, 1, 2], [3, 4, 99]])  # 99 en lugar de 5
        
        metrics_incorrect = ARCMetrics.evaluate_task(pred_incorrect, gt)
        
        # La task NO debe contar como "correcta" (no hay partial credit)
        assert metrics_incorrect['strict_match'] == False, "ARC official metric: single pixel error means task is wrong"
        
        print("✓ Test passed: ARC official metric (strict_match) verified")


def run_all_tests():
    """Ejecuta todos los tests."""
    print("=" * 70)
    print("ARC-AGI-2 METRICS VERIFICATION TESTS")
    print("=" * 70)
    print("\nThese tests verify that the metrics are implemented correctly.")
    print("This is CRITICAL for publication and external verification.\n")
    
    test_suite = TestARCMetrics()
    tests = [
        test_suite.test_strict_match_perfect,
        test_suite.test_strict_match_one_pixel_off,
        test_suite.test_strict_match_different_size,
        test_suite.test_pixel_accuracy_perfect,
        test_suite.test_pixel_accuracy_partial,
        test_suite.test_size_accuracy,
        test_suite.test_evaluate_task_complete,
        test_suite.test_aggregate_metrics,
        test_suite.test_save_and_load_predictions,
        test_suite.test_arc_official_metric,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ Test failed: {test.__name__}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ Test error: {test.__name__}")
            print(f"  Exception: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("Metrics implementation is verified and ready for publication.")
        print("=" * 70)
        return True
    else:
        print(f"\n⚠️  {failed} TESTS FAILED!")
        print("Metrics implementation has issues - DO NOT PUBLISH until fixed.")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
