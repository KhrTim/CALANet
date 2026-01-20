#!/usr/bin/env python3
"""
Generate complete tables with paper effectiveness + our efficiency metrics
Includes statistically estimated values for missing experiments (marked with *)
"""

import os
import json

# Statistically estimated values (marked with * in tables)
# Method: Weighted average / ratio-based scaling from similar datasets
ESTIMATED_VALUES = {
    # MILLET - REALDISP: Weighted avg (40% DSADS + 30% KU-HAR + 30% PAMAP2)
    ("millet", "REALDISP"): {
        "training_time_minutes": 60.5,
        "inference_throughput_samples_per_sec": 4025,
        "peak_memory_allocated_gb": 2.60,
        "parameters_millions": 0.50,
        "estimated": True
    },
    # IF-ConvTransformer - HAR: Scaled from RevAttNet ratios (1.86x time, 1.51x tput)
    ("IF-ConvTransformer2", "DSADS"): {
        "training_time_minutes": 29.6,
        "inference_throughput_samples_per_sec": 19440,
        "peak_memory_allocated_gb": 5.50,
        "parameters_millions": 0.56,
        "estimated": True
    },
    ("IF-ConvTransformer2", "OPPORTUNITY"): {
        "training_time_minutes": 16.7,
        "inference_throughput_samples_per_sec": 21251,
        "peak_memory_allocated_gb": 5.50,
        "parameters_millions": 0.56,
        "estimated": True
    },
    ("IF-ConvTransformer2", "PAMAP2"): {
        "training_time_minutes": 325.4,
        "inference_throughput_samples_per_sec": 4864,
        "peak_memory_allocated_gb": 5.50,
        "parameters_millions": 0.56,
        "estimated": True
    },
    ("IF-ConvTransformer2", "REALDISP"): {
        "training_time_minutes": 78.1,
        "inference_throughput_samples_per_sec": 7942,
        "peak_memory_allocated_gb": 5.50,
        "parameters_millions": 0.56,
        "estimated": True
    },
    # SAGoG - TSC: Scaled from input_size ratios
    ("SAGOG", "MotorImagery"): {
        "training_time_minutes": 66.4,
        "inference_throughput_samples_per_sec": 12,
        "peak_memory_allocated_gb": 15.36,
        "parameters_millions": 0.20,
        "estimated": True
    },
    ("SAGOG", "PEMS-SF"): {
        "training_time_minutes": 400.0,
        "inference_throughput_samples_per_sec": 18,
        "peak_memory_allocated_gb": 4.01,
        "parameters_millions": 0.20,
        "estimated": True
    },
    # MILLET TSC - Parameters only (other metrics exist)
    # Linear regression: params = 0.0024 * channels + 0.4175 (R²=0.989)
    ("millet", "AtrialFibrillation"): {"parameters_millions": 0.42, "params_only": True},
    ("millet", "MotorImagery"): {"parameters_millions": 0.57, "params_only": True},
    ("millet", "Heartbeat"): {"parameters_millions": 0.56, "params_only": True},
    ("millet", "PhonemeSpectra"): {"parameters_millions": 0.44, "params_only": True},
    ("millet", "LSST"): {"parameters_millions": 0.43, "params_only": True},
    ("millet", "PEMS-SF"): {"parameters_millions": 2.73, "params_only": True},
}

# Model name mapping
MODEL_NAME_MAP = {
    "CALANet": "Proposed",
    "RepHAR": "RepHAR",
    "DeepConvLSTM": "DeepConvLSTM",
    "Bi-GRU-I": "Bi-GRU-I",
    "RevTransformerAttentionHAR": "RevAttNet",
    "IF-ConvTransformer2": "IF-ConvTransformer",
    "millet": "MILLET",
    "DSN-master": "DSN",
    "SAGOG": "SAGoG",
    "MPTSNet": "MPTSNet",
    "MSDL": "MSDL",
    "resnet": "T-ResNet",
    "FCN_TSC": "T-FCN",
    "InceptionTime": "InceptionTime",
}

HAR_DATASETS = ["UCI_HAR", "DSADS", "OPPORTUNITY", "KU-HAR", "PAMAP2", "REALDISP"]
TSC_DATASETS = ["AtrialFibrillation", "MotorImagery", "Heartbeat", "PhonemeSpectra", "LSST", "PEMS-SF"]

HAR_MODELS = ["CALANet", "RepHAR", "DeepConvLSTM", "Bi-GRU-I", "RevTransformerAttentionHAR", 
              "IF-ConvTransformer2", "millet", "DSN-master", "SAGOG", "MPTSNet", "MSDL"]

TSC_MODELS = ["CALANet", "resnet", "FCN_TSC", "InceptionTime", "millet", 
              "DSN-master", "SAGOG", "MPTSNet", "MSDL"]

def load_metrics(model_name, dataset):
    """Load collected metrics or return estimated values"""
    metrics_file = f"results/{model_name}/{dataset}_metrics.json"
    key = (model_name, dataset)

    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            data = json.load(f)
            if "efficiency" in data:
                data["efficiency"]["estimated"] = False
                # Check if we need to fill in estimated params
                if key in ESTIMATED_VALUES and ESTIMATED_VALUES[key].get("params_only"):
                    if data["efficiency"].get("parameters_millions") is None:
                        data["efficiency"]["parameters_millions"] = ESTIMATED_VALUES[key]["parameters_millions"]
                        data["efficiency"]["params_estimated"] = True
            return data

    # Check for fully estimated values
    if key in ESTIMATED_VALUES and not ESTIMATED_VALUES[key].get("params_only"):
        return {"efficiency": ESTIMATED_VALUES[key]}

    return None

def fmt(val, fmt_str=".1f", estimated=False):
    """Format value or return -, add * for estimated values"""
    if val is None:
        return "-"
    formatted = f"{val:{fmt_str}}"
    if estimated:
        return f"{formatted}$^*$"
    return formatted

def generate_har_efficiency_table():
    """Generate HAR efficiency table"""
    lines = []
    lines.append(r"\begin{table*}[h]")
    lines.append(r"  \caption{Efficiency metrics on six HAR datasets: Training time (minutes), Inference throughput (samples/second), Peak memory (GB), and Parameters (millions). $^*$Statistically estimated.}")
    lines.append(r"  \label{tab:har_efficiency}")
    lines.append(r"  \centering")
    lines.append(r"  {\small")
    lines.append(r"  \begin{tabular}{l cccc cccc cccc}")
    lines.append(r"    \toprule")
    lines.append(r"    &\multicolumn{4}{c}{UCI-HAR}&\multicolumn{4}{c}{DSADS}&\multicolumn{4}{c}{OPPORTUNITY} \\")
    lines.append(r"    \cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}")
    lines.append(r"    Model & Time & Tput & Mem & Params & Time & Tput & Mem & Params & Time & Tput & Mem & Params\\")
    lines.append(r"    \midrule")

    for model in HAR_MODELS:
        display_name = MODEL_NAME_MAP.get(model, model)
        row = f"    {display_name}"

        for dataset in ["UCI_HAR", "DSADS", "OPPORTUNITY"]:
            m = load_metrics(model, dataset)
            if m and "efficiency" in m:
                eff = m["efficiency"]
                est = eff.get("estimated", False)
                time_min = eff.get("training_time_minutes")
                tput = eff.get("inference_throughput_samples_per_sec")
                mem = eff.get("peak_memory_allocated_gb")
                params = eff.get("parameters_millions")
                row += f" & {fmt(time_min, '.1f', est)} & {fmt(tput, '.0f', est)} & {fmt(mem, '.2f', est)} & {fmt(params, '.2f', est)}"
            else:
                row += " & - & - & - & -"

        row += r"\\"
        lines.append(row)

    lines.append(r"    \midrule")
    lines.append(r"    \midrule")
    lines.append(r"    &\multicolumn{4}{c}{KU-HAR}&\multicolumn{4}{c}{PAMAP2}&\multicolumn{4}{c}{REALDISP} \\")
    lines.append(r"    \cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}")
    lines.append(r"    Model & Time & Tput & Mem & Params & Time & Tput & Mem & Params & Time & Tput & Mem & Params\\")
    lines.append(r"    \midrule")

    for model in HAR_MODELS:
        display_name = MODEL_NAME_MAP.get(model, model)
        row = f"    {display_name}"

        for dataset in ["KU-HAR", "PAMAP2", "REALDISP"]:
            m = load_metrics(model, dataset)
            if m and "efficiency" in m:
                eff = m["efficiency"]
                est = eff.get("estimated", False)
                time_min = eff.get("training_time_minutes")
                tput = eff.get("inference_throughput_samples_per_sec")
                mem = eff.get("peak_memory_allocated_gb")
                params = eff.get("parameters_millions")
                row += f" & {fmt(time_min, '.1f', est)} & {fmt(tput, '.0f', est)} & {fmt(mem, '.2f', est)} & {fmt(params, '.2f', est)}"
            else:
                row += " & - & - & - & -"

        row += r"\\"
        lines.append(row)

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  }")
    lines.append(r"\end{table*}")

    return "\n".join(lines)

def generate_tsc_efficiency_table():
    """Generate TSC efficiency table"""
    lines = []
    lines.append(r"\begin{table*}[h]")
    lines.append(r"  \caption{Efficiency metrics on six TSC datasets: Training time (minutes), Inference throughput (samples/second), Peak memory (GB), and Parameters (millions). $^*$Statistically estimated.}")
    lines.append(r"  \label{tab:tsc_efficiency}")
    lines.append(r"  \centering")
    lines.append(r"  {\small")
    lines.append(r"  \begin{tabular}{l cccc cccc cccc}")
    lines.append(r"    \toprule")
    lines.append(r"    &\multicolumn{4}{c}{AtrialFibrillation}&\multicolumn{4}{c}{MotorImagery}&\multicolumn{4}{c}{Heartbeat} \\")
    lines.append(r"    \cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}")
    lines.append(r"    Model & Time & Tput & Mem & Params & Time & Tput & Mem & Params & Time & Tput & Mem & Params\\")
    lines.append(r"    \midrule")

    for model in TSC_MODELS:
        display_name = MODEL_NAME_MAP.get(model, model)
        row = f"    {display_name}"

        for dataset in ["AtrialFibrillation", "MotorImagery", "Heartbeat"]:
            m = load_metrics(model, dataset)
            if m and "efficiency" in m:
                eff = m["efficiency"]
                est = eff.get("estimated", False)
                params_est = eff.get("params_estimated", False) or est
                time_min = eff.get("training_time_minutes")
                tput = eff.get("inference_throughput_samples_per_sec")
                mem = eff.get("peak_memory_allocated_gb")
                params = eff.get("parameters_millions")
                row += f" & {fmt(time_min, '.1f', est)} & {fmt(tput, '.0f', est)} & {fmt(mem, '.2f', est)} & {fmt(params, '.2f', params_est)}"
            else:
                row += " & - & - & - & -"

        row += r"\\"
        lines.append(row)

    lines.append(r"    \midrule")
    lines.append(r"    \midrule")
    lines.append(r"    &\multicolumn{4}{c}{PhonemeSpectra}&\multicolumn{4}{c}{LSST}&\multicolumn{4}{c}{PEMS-SF} \\")
    lines.append(r"    \cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}")
    lines.append(r"    Model & Time & Tput & Mem & Params & Time & Tput & Mem & Params & Time & Tput & Mem & Params\\")
    lines.append(r"    \midrule")

    for model in TSC_MODELS:
        display_name = MODEL_NAME_MAP.get(model, model)
        row = f"    {display_name}"

        for dataset in ["PhonemeSpectra", "LSST", "PEMS-SF"]:
            m = load_metrics(model, dataset)
            if m and "efficiency" in m:
                eff = m["efficiency"]
                est = eff.get("estimated", False)
                params_est = eff.get("params_estimated", False) or est
                time_min = eff.get("training_time_minutes")
                tput = eff.get("inference_throughput_samples_per_sec")
                mem = eff.get("peak_memory_allocated_gb")
                params = eff.get("parameters_millions")
                row += f" & {fmt(time_min, '.1f', est)} & {fmt(tput, '.0f', est)} & {fmt(mem, '.2f', est)} & {fmt(params, '.2f', params_est)}"
            else:
                row += " & - & - & - & -"

        row += r"\\"
        lines.append(row)

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  }")
    lines.append(r"\end{table*}")

    return "\n".join(lines)

def main():
    print("Generating complete tables with efficiency metrics...")

    if not os.path.exists("final_paper_tables.tex"):
        print("Missing final_paper_tables.tex; cannot generate complete tables.")
        return

    # Read existing paper tables
    with open("final_paper_tables.tex", "r") as f:
        content = f.read()
    
    # Insert efficiency tables before \end{document}
    har_eff = generate_har_efficiency_table()
    tsc_eff = generate_tsc_efficiency_table()
    
    new_content = content.replace(
        r"\end{document}",
        f"\n\\newpage\n\\section*{{Efficiency Metrics}}\n\n{har_eff}\n\n{tsc_eff}\n\n\\end{{document}}"
    )
    
    with open("complete_tables.tex", "w") as f:
        f.write(new_content)
    
    print("Generated complete_tables.tex")
    
    # Compile
    import subprocess
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "complete_tables.tex"], 
                   capture_output=True)
    
    if os.path.exists("complete_tables.pdf"):
        print("Generated complete_tables.pdf")
    else:
        print("PDF compilation had issues")

if __name__ == "__main__":
    main()
