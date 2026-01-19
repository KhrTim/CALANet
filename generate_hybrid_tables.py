#!/usr/bin/env python3
"""
Generate hybrid tables for reviewer response:
- F1/Accuracy: From paper (authoritative)
- Precision/Recall: From our collected data
- Efficiency metrics: From our collected data
- Statistical tests: Based on paper F1/Accuracy values
"""

import os
import json
import numpy as np
from scipy import stats
from paper_results import PAPER_HAR_F1, PAPER_TSC_ACCURACY

# Model name mapping
MODEL_NAME_MAP = {
    "CALANet": "CALANet",
    "RepHAR": "RepHAR",
    "DeepConvLSTM": "DeepConvLSTM",
    "Bi-GRU-I": "Bi-GRU-I",
    "RevTransformerAttentionHAR": "TrAttHAR",
    "IF-ConvTransformer2": "IF-ConvTr",
    "millet": "MILLET",
    "DSN-master": "DSN",
    "DSN": "DSN",
    "SAGOG": "SAGoG",
    "MPTSNet": "MPTSNet",
    "MSDL": "MSDL",
    "resnet": "ResNet",
    "FCN_TSC": "FCN",
    "InceptionTime": "InceptionTime",
}

HAR_DATASETS = ["UCI_HAR", "DSADS", "OPPORTUNITY", "KU-HAR", "PAMAP2", "REALDISP"]
TSC_DATASETS = ["AtrialFibrillation", "Heartbeat", "LSST", "MotorImagery", "PEMS-SF", "PhonemeSpectra"]

HAR_MODELS = ["CALANet", "millet", "RepHAR", "DeepConvLSTM", "Bi-GRU-I",
              "RevTransformerAttentionHAR", "IF-ConvTransformer2", "DSN-master",
              "SAGOG", "MPTSNet", "MSDL"]

TSC_MODELS = ["CALANet", "millet", "resnet", "FCN_TSC", "InceptionTime",
              "DSN-master", "SAGOG", "MPTSNet", "MSDL"]

def load_collected_metrics(model_name, dataset):
    """Load our collected metrics"""
    metrics_file = f"results/{model_name}/{dataset}_metrics.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None

def get_paper_f1(model_name, dataset):
    """Get F1 from paper results"""
    paper_name = model_name
    if model_name == "DSN-master":
        paper_name = "DSN"
    if paper_name in PAPER_HAR_F1 and dataset in PAPER_HAR_F1[paper_name]:
        return PAPER_HAR_F1[paper_name][dataset]
    return None

def get_paper_accuracy(model_name, dataset):
    """Get accuracy from paper results"""
    paper_name = model_name
    if model_name == "DSN-master":
        paper_name = "DSN"
    if model_name == "FCN_TSC":
        paper_name = "FCN"
    if paper_name in PAPER_TSC_ACCURACY and dataset in PAPER_TSC_ACCURACY[paper_name]:
        return PAPER_TSC_ACCURACY[paper_name][dataset]
    return None

def wilcoxon_test(values1, values2, alpha=0.10):
    """Perform Wilcoxon signed-rank test"""
    try:
        pairs = [(v1, v2) for v1, v2 in zip(values1, values2) if v1 is not None and v2 is not None]
        if len(pairs) < 3:
            return None, None
        v1 = [p[0] for p in pairs]
        v2 = [p[1] for p in pairs]
        stat, p_value = stats.wilcoxon(v1, v2)
        mean_diff = np.mean(v1) - np.mean(v2)
        if p_value < alpha:
            if mean_diff > 0:
                return "better", p_value
            else:
                return "worse", p_value
        return "ns", p_value
    except:
        return None, None

def generate_har_effectiveness_table():
    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{HAR Effectiveness Metrics. F1-weighted (\%) from paper; Precision/Recall from experiments. $\blacktriangledown$/$\vartriangle$ indicates significantly worse/better than CALANet ($p<0.10$).}")
    lines.append(r"\label{tab:har_effectiveness}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{l" + "ccc" * len(HAR_DATASETS) + "}")
    lines.append(r"\toprule")

    header = "Model"
    for ds in HAR_DATASETS:
        ds_short = ds.replace("_", "-")
        header += f" & \\multicolumn{{3}}{{c}}{{{ds_short}}}"
    header += r" \\"
    lines.append(header)

    subheader = ""
    for ds in HAR_DATASETS:
        subheader += " & F1 & Prec & Rec"
    subheader += r" \\"
    lines.append(subheader)
    lines.append(r"\midrule")

    calanet_f1s = [get_paper_f1("CALANet", ds) for ds in HAR_DATASETS]

    for model in HAR_MODELS:
        display_name = MODEL_NAME_MAP.get(model, model)
        model_f1s = [get_paper_f1(model, ds) for ds in HAR_DATASETS]

        sig_marker = ""
        if model != "CALANet":
            result, p = wilcoxon_test(model_f1s, calanet_f1s)
            if result == "better":
                sig_marker = r" $\vartriangle$"
            elif result == "worse":
                sig_marker = r" $\blacktriangledown$"

        row = f"{display_name}{sig_marker}"

        for dataset in HAR_DATASETS:
            f1 = get_paper_f1(model, dataset)
            f1_str = f"{f1:.1f}" if f1 is not None else "-"

            metrics = load_collected_metrics(model, dataset)
            if metrics and "effectiveness" in metrics:
                prec = metrics["effectiveness"].get("precision_weighted", None)
                rec = metrics["effectiveness"].get("recall_weighted", None)
                prec_str = f"{prec*100:.1f}" if prec is not None else "-"
                rec_str = f"{rec*100:.1f}" if rec is not None else "-"
            else:
                prec_str = "-"
                rec_str = "-"

            row += f" & {f1_str} & {prec_str} & {rec_str}"

        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)

def generate_tsc_effectiveness_table():
    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{TSC Effectiveness Metrics. Accuracy (\%) from paper; Precision/Recall from experiments. $\blacktriangledown$/$\vartriangle$ indicates significantly worse/better than CALANet ($p<0.10$).}")
    lines.append(r"\label{tab:tsc_effectiveness}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{l" + "ccc" * len(TSC_DATASETS) + "}")
    lines.append(r"\toprule")

    header = "Model"
    for ds in TSC_DATASETS:
        ds_short = ds[:8] if len(ds) > 8 else ds
        header += f" & \\multicolumn{{3}}{{c}}{{{ds_short}}}"
    header += r" \\"
    lines.append(header)

    subheader = ""
    for ds in TSC_DATASETS:
        subheader += " & Acc & Prec & Rec"
    subheader += r" \\"
    lines.append(subheader)
    lines.append(r"\midrule")

    calanet_accs = [get_paper_accuracy("CALANet", ds) for ds in TSC_DATASETS]

    for model in TSC_MODELS:
        display_name = MODEL_NAME_MAP.get(model, model)
        model_accs = [get_paper_accuracy(model, ds) for ds in TSC_DATASETS]

        sig_marker = ""
        if model != "CALANet":
            result, p = wilcoxon_test(model_accs, calanet_accs)
            if result == "better":
                sig_marker = r" $\vartriangle$"
            elif result == "worse":
                sig_marker = r" $\blacktriangledown$"

        row = f"{display_name}{sig_marker}"

        for dataset in TSC_DATASETS:
            acc = get_paper_accuracy(model, dataset)
            acc_str = f"{acc:.1f}" if acc is not None else "-"

            metrics = load_collected_metrics(model, dataset)
            if metrics and "effectiveness" in metrics:
                prec = metrics["effectiveness"].get("precision_weighted", None)
                rec = metrics["effectiveness"].get("recall_weighted", None)
                prec_str = f"{prec*100:.1f}" if prec is not None else "-"
                rec_str = f"{rec*100:.1f}" if rec is not None else "-"
            else:
                prec_str = "-"
                rec_str = "-"

            row += f" & {acc_str} & {prec_str} & {rec_str}"

        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)

def generate_efficiency_table(task="HAR"):
    if task == "HAR":
        models = HAR_MODELS
        datasets = HAR_DATASETS
    else:
        models = TSC_MODELS
        datasets = TSC_DATASETS

    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{task} Efficiency: Training time (min), Throughput (samples/sec), Memory (GB), Parameters (M).}}")
    lines.append(f"\\label{{tab:{task.lower()}_efficiency}}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{l" + "cccc" * len(datasets) + "}")
    lines.append(r"\toprule")

    header = "Model"
    for ds in datasets:
        ds_short = ds.replace("_", "-")[:8]
        header += f" & \\multicolumn{{4}}{{c}}{{{ds_short}}}"
    header += r" \\"
    lines.append(header)

    subheader = ""
    for ds in datasets:
        subheader += " & Time & Tput & Mem & Params"
    subheader += r" \\"
    lines.append(subheader)
    lines.append(r"\midrule")

    for model in models:
        display_name = MODEL_NAME_MAP.get(model, model)
        row = display_name

        for dataset in datasets:
            metrics = load_collected_metrics(model, dataset)
            if metrics and "efficiency" in metrics:
                eff = metrics["efficiency"]
                time_min = eff.get("training_time_minutes", None)
                throughput = eff.get("inference_throughput_samples_per_sec", None)
                memory = eff.get("peak_memory_allocated_gb", None)
                params = eff.get("parameters_millions", None)

                time_str = f"{time_min:.1f}" if time_min is not None else "-"
                tput_str = f"{throughput:.0f}" if throughput is not None else "-"
                mem_str = f"{memory:.2f}" if memory is not None else "-"
                params_str = f"{params:.2f}" if params is not None else "-"
            else:
                time_str = tput_str = mem_str = params_str = "-"

            row += f" & {time_str} & {tput_str} & {mem_str} & {params_str}"

        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)

def main():
    print("Generating hybrid tables...")

    latex = []
    latex.append(r"\documentclass{article}")
    latex.append(r"\usepackage{booktabs}")
    latex.append(r"\usepackage{multirow}")
    latex.append(r"\usepackage{graphicx}")
    latex.append(r"\usepackage{amssymb}")
    latex.append(r"\usepackage[margin=0.5in]{geometry}")
    latex.append(r"\begin{document}")
    latex.append("")
    latex.append(r"\section*{Comprehensive Evaluation Tables}")
    latex.append(r"\textbf{Note:} F1-weighted (HAR) and Accuracy (TSC) values are from the published paper.")
    latex.append(r"Precision, Recall, and efficiency metrics are from our experimental runs.")
    latex.append(r"Statistical significance tests use Wilcoxon signed-rank test at $\alpha=0.10$.")
    latex.append("")

    latex.append(r"\subsection*{Human Activity Recognition (HAR)}")
    latex.append(generate_har_effectiveness_table())
    latex.append("")
    latex.append(generate_efficiency_table("HAR"))
    latex.append("")

    latex.append(r"\subsection*{Time Series Classification (TSC)}")
    latex.append(generate_tsc_effectiveness_table())
    latex.append("")
    latex.append(generate_efficiency_table("TSC"))
    latex.append("")

    latex.append(r"\end{document}")

    with open("hybrid_tables.tex", "w") as f:
        f.write("\n".join(latex))

    print("Generated hybrid_tables.tex")

    import subprocess
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "hybrid_tables.tex"],
        capture_output=True, text=True
    )

    if os.path.exists("hybrid_tables.pdf"):
        print("Generated hybrid_tables.pdf")
    else:
        print("PDF compilation had issues")

if __name__ == "__main__":
    main()
