import json
import os
from typing import List, Dict, Any

def generate_markdown_table(json_path: str, output_path: str):
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    if not data:
        print("Error: No data in JSON.")
        return

    # Extract all config keys found across all entries to build columns
    config_keys = set()
    for entry in data:
        if "config" in entry:
            config_keys.update(entry["config"].keys())
    
    config_keys = sorted(list(config_keys))
    
    headers = config_keys + ["Accuracy", "VRAM (MB)", "Peak VRAM (MB)", "Params", "Duration (s)", "Loss"]
    
    rows = []
    for entry in data:
        if "error" in entry:
            config_vals = [entry["config"].get(k, "-") for k in config_keys]
            rows.append(config_vals + ["ERROR", "-", "-", "-", "-", "-"])
            continue
            
        config_vals = [entry["config"].get(k, "-") for k in config_keys]
        metrics = [
            f"{entry.get('accuracy', 0.0)*100:.1f}%",
            f"{entry.get('vram_mb', 0.0):.2f}",
            f"{entry.get('peak_vram_mb', 0.0):.2f}",
            f"{entry.get('parameters', 0):,}",
            f"{entry.get('duration_sec', 0.0):.2f}",
            f"{entry.get('loss_final', 0.0):.4f}"
        ]
        rows.append(config_vals + metrics)

    # Sort rows by Accuracy (desc) then Duration (asc)
    rows.sort(key=lambda x: (x[len(config_keys)], -float(x[-2]) if x[-2] != "-" else 0), reverse=True)

    with open(output_path, "w") as f:
        f.write(f"# GFN Matrix Results: {data[0].get('task', 'Unknown')}\n\n")
        
        header_row = "| " + " | ".join(headers) + " |"
        sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        f.write(header_row + "\n")
        f.write(sep_row + "\n")
        
        for row in rows:
            f.write("| " + " | ".join([str(v) for v in row]) + " |\n")

    print(f"[MATRIX] Markdown report generated: {output_path}")

if __name__ == "__main__":
    generate_markdown_table(
        "tests/benchmarks/matrix/matrix_XOR_Logic.json",
        "tests/benchmarks/matrix/results_xor.md"
    )
    generate_markdown_table(
        "tests/benchmarks/matrix/matrix_Arithmetic_Sum.json",
        "tests/benchmarks/matrix/results_arithmetic.md"
    )
    generate_markdown_table(
        "tests/benchmarks/matrix/matrix_Hierarchical_Tree.json",
        "tests/benchmarks/matrix/results_hierarchical.md"
    )
    generate_markdown_table(
        "tests/benchmarks/matrix/matrix_Gating_Stability.json",
        "tests/benchmarks/matrix/results_gating.md"
    )
    generate_markdown_table(
        "tests/benchmarks/matrix/matrix_Mini_NLP.json",
        "tests/benchmarks/matrix/results_nlp.md"
    )
    generate_markdown_table(
        "tests/benchmarks/matrix/matrix_Structural_ROI.json",
        "tests/benchmarks/matrix/results_structural.md"
    )
    generate_markdown_table(
        "tests/benchmarks/matrix/matrix_Pooling_ROI.json",
        "tests/benchmarks/matrix/results_pooling.md"
    )
