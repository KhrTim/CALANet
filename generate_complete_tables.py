#!/usr/bin/env python3
"""
Generate complete tables with paper effectiveness + our efficiency metrics
"""

import os
import json

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
    """Load collected metrics"""
    metrics_file = f"results/{model_name}/{dataset}_metrics.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None

def fmt(val, fmt_str=".1f"):
    """Format value or return -"""
    if val is None:
        return "-"
    return f"{val:{fmt_str}}"

def generate_har_efficiency_table():
    """Generate HAR efficiency table"""
    lines = []
    lines.append(r"\begin{table*}[h]")
    lines.append(r"  \caption{Efficiency metrics on six HAR datasets: Training time (minutes), Inference throughput (samples/second), Peak memory (GB), and Parameters (millions).}")
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
                time_min = eff.get("training_time_minutes")
                tput = eff.get("inference_throughput_samples_per_sec")
                mem = eff.get("peak_memory_allocated_gb")
                params = eff.get("parameters_millions")
                row += f" & {fmt(time_min)} & {fmt(tput, '.0f')} & {fmt(mem, '.2f')} & {fmt(params, '.2f')}"
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
                time_min = eff.get("training_time_minutes")
                tput = eff.get("inference_throughput_samples_per_sec")
                mem = eff.get("peak_memory_allocated_gb")
                params = eff.get("parameters_millions")
                row += f" & {fmt(time_min)} & {fmt(tput, '.0f')} & {fmt(mem, '.2f')} & {fmt(params, '.2f')}"
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
    lines.append(r"  \caption{Efficiency metrics on six TSC datasets: Training time (minutes), Inference throughput (samples/second), Peak memory (GB), and Parameters (millions).}")
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
                time_min = eff.get("training_time_minutes")
                tput = eff.get("inference_throughput_samples_per_sec")
                mem = eff.get("peak_memory_allocated_gb")
                params = eff.get("parameters_millions")
                row += f" & {fmt(time_min)} & {fmt(tput, '.0f')} & {fmt(mem, '.2f')} & {fmt(params, '.2f')}"
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
                time_min = eff.get("training_time_minutes")
                tput = eff.get("inference_throughput_samples_per_sec")
                mem = eff.get("peak_memory_allocated_gb")
                params = eff.get("parameters_millions")
                row += f" & {fmt(time_min)} & {fmt(tput, '.0f')} & {fmt(mem, '.2f')} & {fmt(params, '.2f')}"
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
