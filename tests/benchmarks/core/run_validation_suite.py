"""
Master GFN Validation Suite (v2.6.5)
=====================================

Orchestrates the execution of all professional benchmarks.
- Automated discovery and execution
- Structured result aggregation
- Rich visual reporting
"""

import subprocess
import sys
import json
import argparse
import os
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Path Setup
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "tests/benchmarks/core"
RESULTS_BASE = PROJECT_ROOT / "tests/results/core"

console = Console()

BENCHMARKS = {
    'baseline': {
        'script': 'benchmark_baseline_comparison.py',
        'desc': 'Systematic comparison vs RNNs (GRU/LSTM)'
    },
    'composition': {
        'script': 'benchmark_composition.py',
        'desc': 'Function composition & systematic generalization'
    },
    'ablation': {
        'script': 'benchmark_feature_ablation.py',
        'desc': 'Physics feature value-add audit'
    },
    'integrators': {
        'script': 'benchmark_integrators.py',
        'desc': 'Numerical Drift & Symplectic Stability'
    },
    'learning': {
        'script': 'benchmark_learning_dynamics.py',
        'desc': 'GFN vs Transformer on Arithmetic'
    },
    'needle': {
        'script': 'benchmark_needle_haystack.py',
        'desc': '1M Token Long-Context Recall'
    },
    'ood': {
        'script': 'benchmark_ood.py',
        'desc': 'Out-of-Distribution Math Generalization'
    },
    'overhead': {
        'script': 'benchmark_overhead.py',
        'desc': 'Physics Engine Computational Cost'
    },
    'performance': {
        'script': 'benchmark_performance.py',
        'desc': 'Throughput & VRAM Scaling Laws'
    },
    'precision': {
        'script': 'benchmark_precision_stability.py',
        'desc': 'Numerical format robustness (FP16/BF16)'
    },
    'efficiency': {
        'script': 'benchmark_sample_efficiency.py',
        'desc': 'Data efficiency vs Transformers'
    },
    'scaling': {
        'script': 'benchmark_scaling.py',
        'desc': 'Model size expansion laws'
    }
}

def run_bench(name, info):
    """Executes a benchmark script."""
    console.print(f"\n[bold cyan]▶ Running: {name.upper()}[/] ([white]{info['desc']}[/])")
    
    script = BENCHMARK_DIR / info['script']
    if not script.exists():
        console.print(f"  [red]Error: Script not found: {script}[/]")
        return False
        
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)
    
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True
        )
        if proc.returncode == 0:
            console.print(f"  [bold green]✓ SUCCESS[/]")
            return True
        else:
            console.print(f"  [bold red]✗ FAILED (Code {proc.returncode})[/]")
            console.print(f"[dim]{proc.stderr[-1000:]}[/]")
            return False
    except Exception as e:
        console.print(f"  [bold red]✗ ERROR: {e}[/]")
        return False

def show_summary():
    """Aggregates results from individual benchmark logs."""
    console.print(f"\n[bold]GFN ARCHITECTURE VALIDATION REPORT[/] (v2.6.5)\n")

    table = Table(title="Benchmarking Coverage", box=None)
    table.add_column("Category")
    table.add_column("Status")
    table.add_column("Artifacts")

    for name, info in BENCHMARKS.items():
        res_path = RESULTS_BASE / name
        status = "[green]RUN[/]" if res_path.exists() else "[dim]PENDING[/]"
        table.add_row(name.capitalize(), status, str(res_path.relative_to(PROJECT_ROOT)) if res_path.exists() else "-")

    console.print(table)
    console.print(f"\n[dim]Master results located in: {RESULTS_BASE}[/]\n")

def main():
    parser = argparse.ArgumentParser(description='Master GFN Validation Suite')
    parser.add_argument('--all', action='store_true', help='Run every benchmark')
    parser.add_argument('--only', nargs='+', help='Run specific benchmarks')
    parser.add_argument('--status', action='store_true', help='Show coverage status')
    
    args = parser.parse_args()
    
    if args.status:
        show_summary()
        return

    to_run = args.only if args.only else (BENCHMARKS.keys() if args.all else [])
    
    if not to_run:
        parser.print_help()
        return

    start_time = time.time()
    results = {}
    
    for name in to_run:
        if name in BENCHMARKS:
            success = run_bench(name, BENCHMARKS[name])
            results[name] = success
            
    # Final Recap
    elapsed = time.time() - start_time
    passed = sum(1 for v in results.values() if v)
    
    console.print("\n" + "="*60)
    console.print(f"[bold yellow]SUITE RECAP[/] | Elapsed: [cyan]{elapsed:.1f}s[/]")
    console.print(f"Passed: [green]{passed}[/] | Failed: [red]{len(results) - passed}[/]")
    console.print("="*60 + "\n")
    
    show_summary()

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
