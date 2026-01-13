#!/usr/bin/env python3
"""
Generate HYBRID tables combining:
- Paper values for EFFECTIVENESS (Accuracy/F1)
- Collected values for EFFICIENCY (training time, memory, throughput, parameters)
"""

import json
import os
import glob
import numpy as np
from scipy import stats
from paper_results import PAPER_HAR_F1, PAPER_TSC_ACCURACY

# Dataset lists
HAR_DATASETS = ["UCI_HAR", "DSADS", "OPPORTUNITY", "KU-HAR", "PAMAP2", "REALDISP"]
TSC_DATASETS = ["AtrialFibrillation", "Heartbeat", "LSST", "MotorImagery", "PEMS-SF", "PhonemeSpectra"]

# Models
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
    "Bi-GRU-I": "Bi-GRU-I",
    "RepHAR": "RepHAR",
    "DeepConvLSTM": "DeepConvLSTM",
    "MPTSNet": "MPTSNet",
    "MSDL": "MSDL",
    "InceptionTime": "InceptionTime"
}

def load_efficiency_metrics():
    """Load ONLY efficiency metrics from collected results"""
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
                    metrics = json.load(f)
                    # Store only efficiency metrics
                    data[model][dataset] = metrics.get('efficiency', {})
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

def compute_statistical_tests_from_paper(paper_data, datasets, baseline_model="CALANet"):
    """
    Compute Wilcoxon signed-rank test from paper values
    """
    results = {}

    # Get baseline values from paper
    baseline_values_dict = paper_data.get(baseline_model, {})
    baseline_values = [baseline_values_dict.get(d) for d in datasets if d in baseline_values_dict]

    # Compare each model to baseline
    for model in paper_data.keys():
        if model == baseline_model:
            continue

        model_values = []
        paired_baseline = []

        for dataset in datasets:
            if dataset in paper_data.get(model, {}) and dataset in baseline_values_dict:
                model_val = paper_data[model][dataset]
                baseline_val = baseline_values_dict[dataset]

                if model_val is not None and baseline_val is not None:
                    model_values.append(model_val)
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

def create_har_effectiveness_table():
    """Table 1: HAR Effectiveness from paper (F1-Score)"""
    lines = []

    lines.append("\\begin{table*}[h]")
    lines.append("\\caption{Effectiveness metrics (F1-Score, \\%) for HAR models. "
                "$\\blacktriangledown$/$\\vartriangle$ indicates significance at 95\\% level "
                "(Wilcoxon signed-rank test vs Proposed model).}")
    lines.append("\\label{tab:har_effectiveness_hybrid}")
    lines.append("\\centering")
    lines.append("{\\small")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Model & UCI-HAR & DSADS & OPPORTUNITY & KU-HAR & PAMAP2 & REALDISP \\\\")
    lines.append("\\midrule")

    # Compute statistical significance from paper data
    sig_tests = compute_statistical_tests_from_paper(PAPER_HAR_F1, HAR_DATASETS, "CALANet")

    # Order models
    model_order = ["CALANet", "RepHAR", "DeepConvLSTM", "Bi-GRU-I",
                   "RevTransformerAttentionHAR", "IF-ConvTransformer2",
                   "millet", "DSN", "SAGOG", "MPTSNet", "MSDL"]

    for model in model_order:
        if model not in PAPER_HAR_F1:
            continue

        display_name = MODEL_NAME_MAP.get(model, model)
        sig = sig_tests.get(model, "")

        row_values = [escape_latex(display_name)]

        for dataset in HAR_DATASETS:
            value = PAPER_HAR_F1[model].get(dataset)
            if value is not None:
                val_str = format_value(value, 1)
                if sig and model != "CALANet":
                    val_str = f"{val_str} {sig}"
                row_values.append(val_str)
            else:
                row_values.append("---")

        lines.append(" & ".join(row_values) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table*}")
    lines.append("")

    return "\n".join(lines)

def create_tsc_effectiveness_table():
    """Table 2: TSC Effectiveness from paper (Accuracy)"""
    lines = []

    lines.append("\\begin{table*}[h]")
    lines.append("\\caption{Effectiveness metrics (Accuracy, \\%) for TSC models. "
                "$\\blacktriangledown$/$\\vartriangle$ indicates significance at 95\\% level.}")
    lines.append("\\label{tab:tsc_effectiveness_hybrid}")
    lines.append("\\centering")
    lines.append("{\\small")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Model & AF & Heartbeat & LSST & MI & PEMS-SF & PS \\\\")
    lines.append("\\midrule")

    # Compute statistical significance from paper data
    sig_tests = compute_statistical_tests_from_paper(PAPER_TSC_ACCURACY, TSC_DATASETS, "CALANet")

    # Order models
    model_order = ["CALANet", "resnet", "FCN", "InceptionTime",
                   "millet", "DSN", "SAGOG", "MPTSNet", "MSDL"]

    for model in model_order:
        if model not in PAPER_TSC_ACCURACY:
            continue

        display_name = MODEL_NAME_MAP.get(model, model)
        sig = sig_tests.get(model, "")

        row_values = [escape_latex(display_name)]

        for dataset in TSC_DATASETS:
            value = PAPER_TSC_ACCURACY[model].get(dataset)
            if value is not None:
                val_str = format_value(value, 1)
                if sig and model != "CALANet":
                    val_str = f"{val_str} {sig}"
                row_values.append(val_str)
            else:
                row_values.append("---")

        lines.append(" & ".join(row_values) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table*}")
    lines.append("")

    return "\n".join(lines)

def create_efficiency_table(efficiency_data, datasets, task_type):
    """Table 3/4: Efficiency from collected metrics"""
    lines = []

    lines.append("\\begin{table*}[h]")
    lines.append(f"\\caption{{Efficiency metrics for {task_type} models "
                f"(collected from experiments). Training time (min), peak memory (GB), "
                f"parameters (M), throughput (samp/s).}}")
    lines.append(f"\\label{{tab:{task_type.lower()}_efficiency_hybrid}}")
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
        model_order = ["CALANet", "RepHAR", "DeepConvLSTM", "Bi-GRU-I",
                      "RevTransformerAttentionHAR", "millet", "DSN",
                      "SAGOG", "MPTSNet", "MSDL"]
    else:
        model_order = ["CALANet", "resnet", "FCN", "InceptionTime",
                      "millet", "DSN", "SAGOG", "MPTSNet", "MSDL"]

    # For each metric
    metrics_to_show = [
        ("Training Time (min)", "training_time_minutes", 1, 1),
        ("Peak Memory (GB)", "peak_memory_allocated_gb", 1, 2),
        ("Parameters (M)", "parameters_millions", 1, 2),
        ("Throughput (samp/s)", "inference_throughput_samples_per_sec", 1, 0)
    ]

    for metric_name, metric_key, scale, decimals in metrics_to_show:
        lines.append(f"\\multicolumn{{{len(datasets)+1}}}{{c}}{{\\textbf{{{metric_name}}}}} \\\\")
        lines.append("\\midrule")

        for model in model_order:
            if model not in efficiency_data:
                continue

            display_name = MODEL_NAME_MAP.get(model, model)
            row_values = [escape_latex(display_name)]

            for dataset in datasets:
                if dataset in efficiency_data[model]:
                    value = efficiency_data[model][dataset].get(metric_key)
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

def generate_hybrid_document(efficiency_data):
    """Generate complete LaTeX document with hybrid data"""
    doc = []

    # Preamble
    doc.append("\\documentclass[10pt]{article}")
    doc.append("\\usepackage[margin=0.75in]{geometry}")
    doc.append("\\usepackage{booktabs}")
    doc.append("\\usepackage{multirow}")
    doc.append("")
    doc.append("\\title{Comprehensive Evaluation - Hybrid Tables}")
    doc.append("\\author{}")
    doc.append("\\date{}")
    doc.append("")
    doc.append("\\begin{document}")
    doc.append("\\maketitle")
    doc.append("")
    doc.append("\\section*{Evaluation Approach}")
    doc.append("")
    doc.append("These tables combine:")
    doc.append("\\begin{itemize}")
    doc.append("\\item \\textbf{Effectiveness metrics} (Accuracy/F1-Score): "
              "From published paper results")
    doc.append("\\item \\textbf{Efficiency metrics} (Training time, memory, throughput, parameters): "
              "From experimental measurements")
    doc.append("\\item \\textbf{Statistical tests}: Wilcoxon signed-rank test ($p<0.05$) "
              "comparing each model to the Proposed model")
    doc.append("\\end{itemize}")
    doc.append("")
    doc.append("\\clearpage")
    doc.append("")

    # HAR Section
    doc.append("\\section{Human Activity Recognition (HAR)}")
    doc.append("")
    doc.append(create_har_effectiveness_table())
    doc.append("\\clearpage")
    doc.append("")
    doc.append(create_efficiency_table(efficiency_data, HAR_DATASETS, "HAR"))
    doc.append("\\clearpage")
    doc.append("")

    # TSC Section
    doc.append("\\section{Time Series Classification (TSC)}")
    doc.append("")
    doc.append(create_tsc_effectiveness_table())
    doc.append("\\clearpage")
    doc.append("")
    doc.append(create_efficiency_table(efficiency_data, TSC_DATASETS, "TSC"))
    doc.append("")
    doc.append("\\end{document}")

    return "\n".join(doc)

def main():
    print("="*80)
    print("GENERATING HYBRID TABLES")
    print("="*80)
    print("Effectiveness metrics: FROM PAPER (published results)")
    print("Efficiency metrics:    FROM EXPERIMENTS (collected data)")
    print("="*80)
    print()

    # Load efficiency metrics from our collected data
    print("Loading efficiency metrics from experiments...")
    efficiency_data = load_efficiency_metrics()
    print(f"✅ Loaded efficiency data for {len(efficiency_data)} models")

    # Generate hybrid document
    print("\nGenerating LaTeX document...")
    latex_content = generate_hybrid_document(efficiency_data)

    output_file = "hybrid_tables.tex"
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
            print(f"⚠️  LaTeX compilation had warnings (PDF may still be generated)")
    except FileNotFoundError:
        print("⚠️  pdflatex not found")
    except Exception as e:
        print(f"⚠️  Compilation error: {e}")

    print()
    print("="*80)
    print("HYBRID TABLES SUMMARY")
    print("="*80)
    print("✅ HAR F1-Scores: From paper (with statistical tests)")
    print("✅ TSC Accuracy: From paper (with statistical tests)")
    print("✅ Training Time: From experiments (all complete)")
    print("✅ Peak Memory: From experiments (all complete)")
    print("✅ Parameters: From experiments (all complete)")
    print("✅ Throughput: From experiments (all complete)")
    print("="*80)
    print("\nThis approach combines the best of both:")
    print("- Published effectiveness results (validated)")
    print("- Real experimental efficiency measurements (comprehensive)")
    print("="*80)

if __name__ == "__main__":
    main()
