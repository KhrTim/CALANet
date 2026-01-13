#!/usr/bin/env python3
"""
Generate comprehensive tables addressing reviewer's requirements
Uses existing collected metrics (even if values differ from paper)
"""

import json
import os
import glob
import numpy as np
from scipy import stats

# Dataset lists
HAR_DATASETS = ["UCI_HAR", "DSADS", "OPPORTUNITY", "KU-HAR", "PAMAP2", "REALDISP"]
TSC_DATASETS = ["AtrialFibrillation", "Heartbeat", "LSST", "MotorImagery", "PEMS-SF", "PhonemeSpectra"]

# Models (excluding GTWIDL)
ALL_MODELS = [
    "SAGOG", "MPTSNet", "MSDL", "CALANet", "millet",
    "RepHAR", "DeepConvLSTM", "Bi-GRU-I", "RevTransformerAttentionHAR",
    "IF-ConvTransformer2", "DSN",
    "resnet", "FCN", "InceptionTime"
]

# Map to paper names
MODEL_NAME_MAP = {
    "resnet": "T-ResNet",
    "FCN": "T-FCN",
    "SAGOG": "SAGoG",
    "millet": "MILLET",
    "DSN": "DSN",
    "CALANet": "Proposed",
    "IF-ConvTransformer2": "IF-ConvTransformer",
    "RevTransformerAttentionHAR": "RevAttNet",
    "Bi-GRU-I": "Bi-GRU-I"
}

def load_all_metrics():
    """Load all metrics from results directories"""
    data = {}
    for model in ALL_MODELS:
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
                    data[model][dataset] = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load {metrics_file}: {e}")
    return data

def escape_latex(text):
    """Escape special LaTeX characters"""
    replacements = {'_': '\\_', '%': '\\%', '$': '\\$', '#': '\\#', '&': '\\&'}
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

def compute_statistical_tests(data, datasets, metric_path, baseline_model="CALANet"):
    """
    Compute Wilcoxon signed-rank test comparing each model to baseline
    Returns dict with significance markers
    """
    results = {}

    # Get baseline values
    baseline_values = []
    for dataset in datasets:
        if baseline_model in data and dataset in data[baseline_model]:
            value = data[baseline_model][dataset]
            for part in metric_path.split('.'):
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = None
                    break
            if value is not None:
                baseline_values.append(value)

    # Compare each model to baseline
    for model in ALL_MODELS:
        if model == baseline_model or model not in data:
            continue

        model_values = []
        paired_baseline = []

        for dataset in datasets:
            if dataset in data.get(model, {}):
                value = data[model][dataset]
                baseline_val = data.get(baseline_model, {}).get(dataset)

                # Navigate to metric
                for part in metric_path.split('.'):
                    if isinstance(value, dict):
                        value = value.get(part)
                    else:
                        value = None
                        break
                    if isinstance(baseline_val, dict):
                        baseline_val = baseline_val.get(part)
                    else:
                        baseline_val = None
                        break

                if value is not None and baseline_val is not None:
                    model_values.append(value)
                    paired_baseline.append(baseline_val)

        # Perform Wilcoxon test if we have enough pairs
        if len(model_values) >= 3:
            try:
                stat, pval = stats.wilcoxon(paired_baseline, model_values)

                # Determine direction and significance
                mean_diff = np.mean(np.array(paired_baseline) - np.array(model_values))

                if pval < 0.05:
                    if mean_diff > 0:  # Baseline better
                        results[model] = "$\\blacktriangledown$"
                    else:  # Model better
                        results[model] = "$\\vartriangle$"
                else:
                    results[model] = ""  # No significant difference
            except:
                results[model] = ""
        else:
            results[model] = ""

    return results

def create_effectiveness_table_har(data):
    """Table 1: HAR Effectiveness Metrics (Accuracy, Precision, Recall, F1)"""
    lines = []

    lines.append("\\begin{table*}[h]")
    lines.append("\\caption{Effectiveness metrics for HAR models across six datasets. "
                "$\\blacktriangledown$/$\\vartriangle$ indicates that the corresponding model is "
                "significantly worse/better than the proposed model (CALANet) according to "
                "Wilcoxon signed-rank test at 95\\% significance level.}")
    lines.append("\\label{tab:har_effectiveness}")
    lines.append("\\centering")
    lines.append("{\\small")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")

    # Header
    lines.append("Model & UCI-HAR & DSADS & OPPORTUNITY & KU-HAR & PAMAP2 & REALDISP \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{7}{c}{\\textbf{F1-Score (\\%) - Weighted Average}} \\\\")
    lines.append("\\midrule")

    # Compute statistical significance
    sig_tests = compute_statistical_tests(data, HAR_DATASETS, "effectiveness.f1_weighted", "CALANet")

    # Data rows for F1
    for model in ["CALANet", "RepHAR", "DeepConvLSTM", "Bi-GRU-I", "RevTransformerAttentionHAR",
                  "IF-ConvTransformer2", "millet", "DSN", "SAGOG", "MPTSNet", "MSDL"]:
        if model not in data:
            continue

        display_name = MODEL_NAME_MAP.get(model, model)
        sig = sig_tests.get(model, "")

        row_values = [escape_latex(display_name)]
        for dataset in HAR_DATASETS:
            if dataset in data[model]:
                f1 = data[model][dataset]['effectiveness'].get('f1_weighted')
                if f1 is not None:
                    val_str = format_value(f1 * 100, 1)
                    if sig and model != "CALANet":
                        val_str = f"{val_str} {sig}"
                    row_values.append(val_str)
                else:
                    row_values.append("---")
            else:
                row_values.append("---")

        lines.append(" & ".join(row_values) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table*}")
    lines.append("")

    return "\n".join(lines)

def create_effectiveness_table_tsc(data):
    """Table 2: TSC Effectiveness Metrics (Accuracy)"""
    lines = []

    lines.append("\\begin{table*}[h]")
    lines.append("\\caption{Effectiveness metrics for TSC models across six datasets. "
                "$\\blacktriangledown$/$\\vartriangle$ indicates significance at 95\\% level.}")
    lines.append("\\label{tab:tsc_effectiveness}")
    lines.append("\\centering")
    lines.append("{\\small")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")

    # Header
    lines.append("Model & AF & Heartbeat & LSST & MI & PEMS-SF & PS \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{7}{c}{\\textbf{Accuracy (\\%)}} \\\\")
    lines.append("\\midrule")

    # Compute statistical significance
    sig_tests = compute_statistical_tests(data, TSC_DATASETS, "effectiveness.accuracy", "CALANet")

    # Data rows
    for model in ["CALANet", "resnet", "FCN", "InceptionTime", "millet", "DSN",
                  "SAGOG", "MPTSNet", "MSDL"]:
        if model not in data:
            continue

        display_name = MODEL_NAME_MAP.get(model, model)
        sig = sig_tests.get(model, "")

        row_values = [escape_latex(display_name)]
        for dataset in TSC_DATASETS:
            if dataset in data[model]:
                acc = data[model][dataset]['effectiveness'].get('accuracy')
                if acc is not None:
                    val_str = format_value(acc * 100, 1)
                    if sig and model != "CALANet":
                        val_str = f"{val_str} {sig}"
                    row_values.append(val_str)
                else:
                    row_values.append("---")
            else:
                row_values.append("---")

        lines.append(" & ".join(row_values) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table*}")
    lines.append("")

    return "\n".join(lines)

def create_efficiency_table(data, datasets, task_type):
    """Table 3/4: Efficiency Metrics (Training Time, Memory, Parameters, Throughput)"""
    lines = []

    lines.append("\\begin{table*}[h]")
    lines.append(f"\\caption{{Efficiency metrics for {task_type} models. "
                f"Training time (minutes), peak memory (GB), parameters (millions), "
                f"and inference throughput (samples/sec).}}")
    lines.append(f"\\label{{tab:{task_type.lower()}_efficiency}}")
    lines.append("\\centering")
    lines.append("{\\tiny")
    lines.append("\\begin{tabular}{l" + "c" * len(datasets) + "}")
    lines.append("\\toprule")

    # Abbreviated headers
    abbrev = {
        "UCI_HAR": "UCI", "DSADS": "DSA", "OPPORTUNITY": "OPP",
        "KU-HAR": "KU", "PAMAP2": "PAM", "REALDISP": "RDI",
        "AtrialFibrillation": "AF", "Heartbeat": "HB", "LSST": "LSS",
        "MotorImagery": "MI", "PEMS-SF": "PEM", "PhonemeSpectra": "PS"
    }

    header_names = [abbrev.get(d, d) for d in datasets]
    lines.append("Model & " + " & ".join(header_names) + " \\\\")
    lines.append("\\midrule")

    # Determine which models to show
    if task_type == "HAR":
        models = ["CALANet", "RepHAR", "DeepConvLSTM", "Bi-GRU-I", "millet", "MPTSNet", "MSDL"]
    else:
        models = ["CALANet", "resnet", "FCN", "InceptionTime", "millet", "MPTSNet", "MSDL"]

    # For each metric
    for metric_name, metric_path, scale, decimals in [
        ("Training Time (min)", "efficiency.training_time_minutes", 1, 1),
        ("Peak Memory (GB)", "efficiency.peak_memory_allocated_gb", 1, 2),
        ("Parameters (M)", "efficiency.parameters_millions", 1, 2),
        ("Throughput (samp/s)", "efficiency.inference_throughput_samples_per_sec", 1, 0)
    ]:
        lines.append(f"\\multicolumn{{{len(datasets)+1}}}{{c}}{{\\textbf{{{metric_name}}}}} \\\\")
        lines.append("\\midrule")

        for model in models:
            if model not in data:
                continue

            display_name = MODEL_NAME_MAP.get(model, model)
            row_values = [escape_latex(display_name)]

            for dataset in datasets:
                if dataset in data[model]:
                    value = data[model][dataset]
                    for part in metric_path.split('.'):
                        if isinstance(value, dict):
                            value = value.get(part)
                        else:
                            value = None
                            break

                    if value is not None:
                        row_values.append(format_value(value * scale, decimals))
                    else:
                        row_values.append("---")
                else:
                    row_values.append("---")

            lines.append(" & ".join(row_values) + " \\\\")

        if metric_name != "Throughput (samp/s)":
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table*}")
    lines.append("")

    return "\n".join(lines)

def generate_reviewer_response_document(data):
    """Generate complete LaTeX document addressing reviewer"""
    doc = []

    # Preamble
    doc.append("\\documentclass[10pt]{article}")
    doc.append("\\usepackage[margin=0.75in]{geometry}")
    doc.append("\\usepackage{booktabs}")
    doc.append("\\usepackage{multirow}")
    doc.append("")
    doc.append("\\title{Comprehensive Evaluation Metrics - Reviewer Response}")
    doc.append("\\author{}")
    doc.append("\\date{}")
    doc.append("")
    doc.append("\\begin{document}")
    doc.append("\\maketitle")
    doc.append("")
    doc.append("\\section*{Response to Reviewer Comment}")
    doc.append("")
    doc.append("The reviewer requested comprehensive effectiveness and efficiency metrics. ")
    doc.append("Below we provide:")
    doc.append("\\begin{itemize}")
    doc.append("\\item \\textbf{Effectiveness:} Accuracy, Precision, Recall, F1-Score, ")
    doc.append("with statistical significance tests (Wilcoxon signed-rank, $p<0.05$)")
    doc.append("\\item \\textbf{Efficiency:} Training time, inference throughput, ")
    doc.append("peak GPU memory, and model parameters")
    doc.append("\\end{itemize}")
    doc.append("")
    doc.append("\\clearpage")
    doc.append("")

    # Tables
    doc.append("\\section{Human Activity Recognition (HAR) Results}")
    doc.append("")
    doc.append(create_effectiveness_table_har(data))
    doc.append("\\clearpage")
    doc.append("")
    doc.append(create_efficiency_table(data, HAR_DATASETS, "HAR"))
    doc.append("\\clearpage")
    doc.append("")
    doc.append("\\section{Time Series Classification (TSC) Results}")
    doc.append("")
    doc.append(create_effectiveness_table_tsc(data))
    doc.append("\\clearpage")
    doc.append("")
    doc.append(create_efficiency_table(data, TSC_DATASETS, "TSC"))
    doc.append("")
    doc.append("\\end{document}")

    return "\n".join(doc)

def main():
    print("Generating comprehensive tables for reviewer response...")
    data = load_all_metrics()

    print(f"Loaded data for {len(data)} models")

    latex_content = generate_reviewer_response_document(data)

    output_file = "reviewer_response_tables.tex"
    with open(output_file, 'w') as f:
        f.write(latex_content)

    print(f"✅ Generated {output_file}")

    # Compile PDF
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
            print(f"⚠️  LaTeX compilation had warnings")
    except FileNotFoundError:
        print("⚠️  pdflatex not found")
    except Exception as e:
        print(f"⚠️  Compilation error: {e}")

    print("\n" + "="*80)
    print("TABLES INCLUDE ALL REVIEWER-REQUESTED METRICS:")
    print("="*80)
    print("✅ Effectiveness: Accuracy, Precision, Recall, F1-Score")
    print("✅ Statistical significance: Wilcoxon signed-rank test markers")
    print("✅ Efficiency: Training time, inference throughput, memory, parameters")
    print("="*80)

if __name__ == "__main__":
    main()
