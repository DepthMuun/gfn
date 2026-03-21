import argparse
import time
from pathlib import Path
from rich.console import Console
from rich.progress import track

from gfn.realizations.gssm.config import ManifoldConfig
from tests.matrix.generator import MatrixGenerator
from tests.matrix.runner import MatrixRunner
from tests.matrix.analyser import MatrixAnalyser

def main():
    parser = argparse.ArgumentParser(description="GFN Matrix Testing Suite")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of trials (for sanity check)")
    parser.add_argument("--filter-integrator", type=str, default=None, help="Filter by integrator")
    args = parser.parse_args()
    
    # Force UTF-8 on Windows
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    
    console = Console()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent.parent / "results" / "matrix" / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold magenta]Starting Matrix Suite ({timestamp})[/]")
    console.print(f"Results Directory: {results_dir}")
    
    # 1. Generate Configs
    console.print("[yellow]Generating Configurations...[/]")
    all_configs = list(MatrixGenerator.generate_all())
    
    # Filter
    if args.filter_integrator:
        all_configs = [c for c in all_configs if c.integrator == args.filter_integrator]
        
    total_configs = len(all_configs)
    
    if args.limit:
        all_configs = all_configs[:args.limit]
        console.print(f"[dim]Limited to {args.limit} trials via --limit[/]")
        
    console.print(f"Scheduled Trials: {len(all_configs)} / {total_configs} total permutations")
    
    # 2. Run Trials with Enhanced Progress
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
    import sys
    import os
    from contextlib import contextmanager

    @contextmanager
    def suppress_stdout():
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout

    runner = MatrixRunner(results_dir)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[status]}"),
        console=console
    ) as progress:
        task_id = progress.add_task("Running Matrix...", total=len(all_configs), status="[Starting]")
        
        for i, config in enumerate(all_configs):
            # Update status with current config details
            short_desc = f"{config.integrator}/{config.physics.topology.type}"
            progress.update(task_id, status=f"[Current: {short_desc}]")
            
            # Run trial (Suppress noisy model prints)
            with suppress_stdout():
                metrics = runner.run_trial(config, i)
            
            # Post-update
            status_color = "green" if metrics.get("status") == "SUCCESS" else "red"
            progress.update(task_id, advance=1, status=f"[{status_color}]Last: {metrics.get('status')}[/]")
        
    # 3. Analyze
    console.print("[green]Matrix Complete. Generating Report...[/]")
    analyser = MatrixAnalyser(results_dir)
    analyser.generate_report()
    
if __name__ == "__main__":
    main()
