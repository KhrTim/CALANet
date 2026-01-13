#!/usr/bin/env python3
"""
Generate LaTeX tables with models as rows and datasets as columns for each metric
"""

import json
import os
import glob
import numpy as np

# Dataset lists
HAR_DATASETS = ["UCI_HAR", "DSADS", "OPPORTUNITY", "KU-HAR", "PAMAP2", "REALDISP"]
TSC_DATASETS = ["AtrialFibrillation", "Heartbeat", "LSST", "MotorImagery", "PEMS-SF", "PhonemeSpectra"]

# All models (excluding GTWIDL)
ALL_MODELS = [
    "SAGOG", "MPTSNet", "MSDL", "CALANet", "millet",
    "RepHAR", "DeepConvLSTM", "Bi-GRU-I", "RevTransformerAttentionHAR",
    "IF-ConvTransformer2", "DSN",
    "resnet", "FCN", "InceptionTime"
]

def load_all_metrics():
    """Load all metrics from results directories"""
    data = {}

    for model in ALL_MODELS:
        # Map model names to directory names
        results_model = "DSN-master" if model == "DSN" else model
        results_model = "FCN_TSC" if model == "FCN" else results_model

        model_dir = f"results/{results_model}"
        if not os.path.exists(model_dir):
            continue

        data[model] = {}

        for metrics_file in glob.glob(f"{model_dir}/*_metrics.json"):
            dataset = os.path.basename(metrics_file).replace('_metrics.json', '')

            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    data[model][dataset] = metrics
            except Exception as e:
                print(f"Warning: Could not load {metrics_file}: {e}")

    return data

def escape_latex(text):
    """Escape special LaTeX characters"""
    replacements = {
        '_': '\\_',
        '%': '\\%',
        '$': '\\$',
        '#': '\\#',
        '&': '\\&',
        '{': '\\{',
        '}': '\\}',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def format_value(value, decimal_places=2):
    """Format a value for LaTeX display"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "---"
    if isinstance(value, (int, float)):
        if decimal_places == 0:
            return f"{value:.0f}"
        return f"{value:.{decimal_places}f}"
    return str(value)

def create_metric_table(data, datasets, metric_path, metric_name, decimal_places=2,
                        scale_factor=1.0, task_type="HAR"):
    """
    Create a LaTeX table for a specific metric

    Args:
        data: Dictionary of all loaded metrics
        datasets: List of datasets to include
        metric_path: Path to metric in JSON (e.g., 'effectiveness.accuracy')
        metric_name: Display name for the metric
        decimal_places: Number of decimal places
        scale_factor: Multiply values by this (e.g., 100 for percentages)
        task_type: HAR or TSC
    """
    lines = []

    # Start table
    num_cols = len(datasets) + 1  # +1 for model name column
    col_spec = "l" + "c" * len(datasets)

    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{metric_name} for {task_type} Models}}")
    lines.append(f"\\label{{tab:{task_type.lower()}_{metric_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}}}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header row
    header = "\\textbf{Model} & " + " & ".join([f"\\textbf{{{escape_latex(d)}}}" for d in datasets]) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Get metric path parts
    path_parts = metric_path.split('.')

    # Data rows - iterate through models that support this task
    for model in ALL_MODELS:
        if model not in data:
            continue

        # Check if model has data for any dataset in this task
        has_data = any(dataset in data[model] for dataset in datasets)
        if not has_data:
            continue

        row_values = [escape_latex(model)]

        for dataset in datasets:
            if dataset in data[model]:
                # Navigate to the metric value
                value = data[model][dataset]
                for part in path_parts:
                    if isinstance(value, dict):
                        value = value.get(part)
                    else:
                        value = None
                        break

                if value is not None:
                    value = value * scale_factor
                    row_values.append(format_value(value, decimal_places))
                else:
                    row_values.append("---")
            else:
                row_values.append("---")

        lines.append(" & ".join(row_values) + " \\\\")

    # End table
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    lines.append("")

    return "\n".join(lines)

def generate_latex_document(data):
    """Generate complete LaTeX document with all metric tables"""

    doc = []

    # Preamble
    doc.append("\\documentclass[10pt]{article}")
    doc.append("\\usepackage[margin=1in]{geometry}")
    doc.append("\\usepackage{booktabs}")
    doc.append("\\usepackage{multirow}")
    doc.append("\\usepackage{rotating}")
    doc.append("\\usepackage{longtable}")
    doc.append("\\usepackage{array}")
    doc.append("")
    doc.append("\\title{Comprehensive Model Performance Metrics by Dataset}")
    doc.append("\\author{}")
    doc.append("\\date{}")
    doc.append("")
    doc.append("\\begin{document}")
    doc.append("\\maketitle")
    doc.append("")
    doc.append("\\section{Human Activity Recognition (HAR) Results}")
    doc.append("")

    # HAR Tables
    har_tables = [
        ("effectiveness.accuracy", "Accuracy (\\%)", 2, 100),
        ("effectiveness.f1_macro", "F1-Score Macro (\\%)", 2, 100),
        ("effectiveness.precision_macro", "Precision Macro (\\%)", 2, 100),
        ("effectiveness.recall_macro", "Recall Macro (\\%)", 2, 100),
        ("efficiency.training_time_seconds", "Training Time (minutes)", 1, 1/60),
        ("efficiency.inference_throughput_samples_per_sec", "Inference Throughput (samples/sec)", 0, 1),
        ("efficiency.total_parameters", "Parameters (millions)", 2, 1e-6),
        ("efficiency.peak_memory_allocated_gb", "Peak Memory (GB)", 2, 1),
    ]

    for metric_path, metric_name, decimal_places, scale_factor in har_tables:
        table = create_metric_table(data, HAR_DATASETS, metric_path, metric_name,
                                   decimal_places, scale_factor, "HAR")
        doc.append(table)
        doc.append("\\clearpage")
        doc.append("")

    doc.append("\\section{Time Series Classification (TSC) Results}")
    doc.append("")

    # TSC Tables
    tsc_tables = [
        ("effectiveness.accuracy", "Accuracy (\\%)", 2, 100),
        ("effectiveness.f1_macro", "F1-Score Macro (\\%)", 2, 100),
        ("effectiveness.precision_macro", "Precision Macro (\\%)", 2, 100),
        ("effectiveness.recall_macro", "Recall Macro (\\%)", 2, 100),
        ("efficiency.training_time_seconds", "Training Time (minutes)", 1, 1/60),
        ("efficiency.inference_throughput_samples_per_sec", "Inference Throughput (samples/sec)", 0, 1),
        ("efficiency.total_parameters", "Parameters (millions)", 2, 1e-6),
        ("efficiency.peak_memory_allocated_gb", "Peak Memory (GB)", 2, 1),
    ]

    for metric_path, metric_name, decimal_places, scale_factor in tsc_tables:
        table = create_metric_table(data, TSC_DATASETS, metric_path, metric_name,
                                   decimal_places, scale_factor, "TSC")
        doc.append(table)
        doc.append("\\clearpage")
        doc.append("")

    doc.append("\\end{document}")

    return "\n".join(doc)

def main():
    print("Loading all metrics...")
    data = load_all_metrics()

    print(f"Loaded data for {len(data)} models")
    for model, datasets in data.items():
        print(f"  {model}: {len(datasets)} datasets")

    print("\nGenerating LaTeX document...")
    latex_content = generate_latex_document(data)

    output_file = "metrics_by_dataset_tables.tex"
    with open(output_file, 'w') as f:
        f.write(latex_content)

    print(f"✅ Generated {output_file}")
    print(f"   Total length: {len(latex_content)} characters")

    print("\nCompiling to PDF...")
    import subprocess
    try:
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', output_file],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            # Run twice for references
            subprocess.run(['pdflatex', '-interaction=nonstopmode', output_file],
                         capture_output=True, timeout=60)
            print(f"✅ Generated {output_file.replace('.tex', '.pdf')}")
        else:
            print(f"⚠️  LaTeX compilation had warnings (PDF may still be generated)")

    except FileNotFoundError:
        print("⚠️  pdflatex not found. Install texlive to compile PDFs.")
    except Exception as e:
        print(f"⚠️  Compilation error: {e}")

    print("\nDone!")

if __name__ == "__main__":
    main()
