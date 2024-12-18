"""
fig. 3 in the paper
"""
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def create_histogram(data, output_path):
    sns.set_theme(style="ticks", font="Arial", font_scale=1.8)

    f, ax = plt.subplots(figsize=(10, 6))
    sns.despine(f)

    hue_order = (
        "Arithmetic Units",
        "Clock Management Units",
        "Communication Protocol Units",
        "Control Logic Units",
        "Data Path Units",
        "Encryption Units",
        "Signal Processing Units",
        "Other Units",
        "Inconsistent"
    )

    total_count = len(data)
    counts = data['Circuit unit type'].value_counts()
    percentages = (counts / total_count * 100).round(1)

    legend_labels = {circuit_type: f"{circuit_type} - {percentages[circuit_type]}%" for circuit_type in hue_order}

    palette = sns.diverging_palette(220, 20, n=len(hue_order), as_cmap=False)

    sns.histplot(
        data,
        x="#node",
        hue="Circuit unit type",
        hue_order=hue_order,
        multiple="stack",
        binwidth=0.2,
        palette=palette,
        edgecolor=".3",
        linewidth=0.1,
        log_scale=True,
    )

    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks([1, 10, 100, 1000, 10000, 100000])
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    ax.set_xlabel("Number of node", fontweight='bold')
    ax.set_ylabel("")

    legend = ax.get_legend()
    legend.set_bbox_to_anchor((1.18, 1.05))
    legend.set_loc('upper right')

    for text in legend.get_texts():
        circuit_type = text.get_text()
        text.set_text(legend_labels[circuit_type])
        plt.setp(text, fontsize='16')

    plt.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight"
    )

def get_data(csv_path, json_path):
    data = pd.read_csv(csv_path)
    with open(json_path, 'r') as f:
        rtl_data = json.load(f)
    rtl_id_to_circuit_type = {int(key): value['consistent_label'] if value['consistent_label'] else "Inconsistent" for key, value in rtl_data.items()}
    data['Circuit unit type'] = data['rtl_id'].map(rtl_id_to_circuit_type)
    return data
