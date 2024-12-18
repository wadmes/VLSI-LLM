"""
fig. 6 in the paper
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib
import numpy as np

def accuracy_comparison(output_path):
    matplotlib.rcParams['font.family'] = 'Arial'
    data = {
        "Model": ["RTL", "Netlist", "RTL", "Netlist", "RTL",  "Netlist", "Netlist", "Netlist", "Netlist"],
        "Accuracy": [78.70, 52.98, 46.85, 32.48, 14.33, 10.24, 59.09, 55.14, 72.97],
        "Type": [
            "GPT-4o", "GPT-4o",
            "Llama3.1-8B", "Llama3.1-8B",
            "Llama3.2-3B", "Llama3.2-3B",
            "BRIDGES_no_lora", "BRIDGES_no_stage1", "BRIDGES"
        ],
    }

    df = pd.DataFrame(data)
    models = df["Model"].unique()
    types = df["Type"].unique()

    full_grid = pd.DataFrame(
        [(model, t) for model in models for t in types],
        columns=["Model", "Type"]
    )
    df = full_grid.merge(df, on=["Model", "Type"], how="left").fillna(0)

    color_map = {
        "GPT-4o": "#74AA9C",  # GPT Green
        "Llama3.1-8B": "#629CD3",  # Llama Blue
        "Llama3.2-3B": "#ADD8E6",  # Llama Blue
        "BRIDGES_no_lora": "#EEB886",  # Light Orange
        "BRIDGES": "#F46A12",  # Dark Orange
        "BRIDGES_no_stage1": "#FF935D",  # Medium Orange
        "white": "#FFFFFF",  # Medium Orange
    }

    p = [[0.3, 0.82], [0.42, 0.94], [0.54, 1.06], [0.4, 1.18], [0.6, 1.30], [0.8, 1.42]]
    bar_width = 0.1

    plt.figure(figsize=(8, 4))

    for i, t in enumerate(types):
        subset = df[df["Type"] == t]
        positions = p[i]
        plt.bar(
            positions,
            subset["Accuracy"],
            width=bar_width,
            label=t,
            color=color_map[t],
            edgecolor="black",
            linewidth=1
        )

    plt.xticks(np.array([0.42, 1.12]), models, fontsize=18, fontname="Arial")
    plt.xlim(0.1, 1.6)
    plt.xlabel("Input level", fontsize=18, fontname="Arial")
    plt.xticks(fontsize=16, fontname="Arial")
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 20))
    plt.ylabel("Accuracy (%)", fontsize=18, fontname="Arial")
    plt.yticks(fontsize=18, fontname="Arial")

    for p in plt.gca().patches:
        height = p.get_height()
        if height > 0:
            plt.text(
                p.get_x() + p.get_width() / 2.,
                height + 1,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=19
            )

    plt.legend(
        handles=[
            Patch(color=color_map["GPT-4o"], label="GPT-4o"),
            Patch(color=color_map["Llama3.1-8B"], label="Llama3-8B"),
            Patch(color=color_map["Llama3.2-3B"], label="Llama3-3B"),
            Patch(color=color_map["BRIDGES_no_lora"], label="BRIDGES_no_lora"),
            Patch(color=color_map["BRIDGES_no_stage1"], label="BRIDGES_no_stage1"),
            Patch(color=color_map["BRIDGES"], label="BRIDGES"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.53, 1.07),
        ncol=2,
        fontsize=15,
        columnspacing=0.5,handletextpad=0.2,
        frameon=False,
        title=None
    )

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_linewidth(1.5)
    plt.gca().spines["bottom"].set_linewidth(1.5)
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()
