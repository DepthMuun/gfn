import json
import os
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
from rich.console import Console
from rich.table import Table

class MatrixAnalyser:
    """
    Aggregates and reports on matrix run results.
    """
    
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.console = Console()
        
    def generate_report(self):
        """Scans the run directory and produces a summary."""
        trials = []
        
        # Scan for trial folders
        for trial_dir in self.run_dir.glob("trial_*"):
            metrics_path = trial_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r", encoding='utf-8') as f:
                    try:
                        metrics = json.load(f)
                        trials.append(metrics)
                    except:
                        pass
        
        if not trials:
            self.console.print("[red]No valid trials found.[/]")
            return
            
        # 1. Overall Stats
        total = len(trials)
        success = sum(1 for t in trials if t.get("status") == "SUCCESS")
        crash = sum(1 for t in trials if t.get("status") == "CRASH")
        
        self.console.print(f"\n[bold]Matrix Run Report[/] ({self.run_dir.name})")
        self.console.print(f"Total Trials: {total} | Success: [green]{success}[/] | Crash: [red]{crash}[/]")
        
        # 2. Top Performers (Fastest Convergence)
        # Filter successes
        success_trials = [t for t in trials if t.get("status") == "SUCCESS"]
        success_trials.sort(key=lambda x: x.get("time", 9999.0))
        
        table = Table(title="Top 10 Fastest Configurations")
        table.add_column("ID", style="cyan")
        table.add_column("Config", style="magenta")
        table.add_column("Time (s)", style="green")
        table.add_column("Steps", style="yellow")
        table.add_column("Loss", style="blue")
        
        for t in success_trials[:10]:
            table.add_row(
                str(t.get("trial_id")),
                t.get("config_summary", "N/A"),
                f"{t.get('time', 0):.4f}",
                str(t.get("steps")),
                f"{t.get('final_loss', 0):.4f}"
            )
            
        self.console.print(table)
        
        # 3. Crash Report
        crash_trials = [t for t in trials if t.get("status") == "CRASH"]
        if crash_trials:
            ctable = Table(title="Crashed Configurations", style="red")
            ctable.add_column("ID", style="cyan")
            ctable.add_column("Config", style="magenta")
            ctable.add_column("Error", style="red")
            
            for t in crash_trials[:5]: # Top 5 crashes
                ctable.add_row(
                    str(t.get("trial_id")),
                    t.get("config_summary", "N/A"),
                    t.get("error", "Unknown")[:50] + "..."
                )
            self.console.print(ctable)
            
        # Save summary JSON
        summary = {
            "total": total,
            "success": success,
            "crash": crash,
            "top_10": success_trials[:10],
            "crashes": crash_trials
        }
        with open(self.run_dir / "summary.json", "w", encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
            
        self.console.print(f"[dim]Report saved to {self.run_dir / 'summary.json'}[/]")
