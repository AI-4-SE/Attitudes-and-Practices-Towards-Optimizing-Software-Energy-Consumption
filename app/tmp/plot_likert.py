import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.patches import Patch
import matplotlib.pyplot as plt

def plot_likert_5point(
    data,
    out_path,
    legend_title=None,
    legend_loc="center right",
    normalize=False,
    category_labels=None,
    category_colors=None,
    text_color="black",
    fontsize=9,
    percentages=False,
    x_axis_lables=None,
    plot_title=""
):
    groups = list(data.keys())
    cats = [1, 2, 3, 4, 5]

    # Raw counts matrix (for printing)
    raw_counts = np.array([[data[g][c] for c in cats] for g in groups], dtype=float)
    row_totals = raw_counts.sum(axis=1, keepdims=True)  # <- for percentages

    # Values used for bar heights (maybe normalized)
    if normalize:
        counts = raw_counts / row_totals * 100
    else:
        counts = raw_counts.copy()

    c1, c2, c3, c4, c5 = counts.T

    # Offsets for centered Likert
    left1 = - (c1 + c2) - c3 / 2
    left2 = - c2 - c3 / 2
    left3 = - c3 / 2
    left4 =  c3 / 2
    left5 =  c3 / 2 + c4

    lefts = [left1, left2, left3, left4, left5]
    widths = [c1, c2, c3, c4, c5]

    y_pos = np.arange(len(groups))

    # Default colors
    default_colors = {
        1: "#b2182b",
        2: "#ef8a62",
        3: "#f7f7f7",
        4: "#67a9cf",
        5: "#2166ac"
    }

    if category_colors is None:
        colors = default_colors
    else:
        colors = {c: category_colors.get(c, default_colors[c]) for c in cats}

    fig_height = 4 if len(data) > 5 else 3
    fig, ax = plt.subplots(figsize=(9, fig_height))

    # 0-line behind everything
    ax.axvline(0, color="black", linewidth=1, linestyle="--", zorder=0)

    

    # Draw bars and labels
    # 
    for idx, cat in enumerate(cats):
        ax.barh(
            y_pos,
            widths[idx],
            left=lefts[idx],
            height=0.6,
            color=colors[cat],
            zorder=10,
            edgecolor = "none",
            linewidth = 0
        )

        # Add absolute + percentage on top: "12 (34%)"
        for row in range(len(groups)):
            val = int(raw_counts[row, idx])
            if val > 0:
                pct = raw_counts[row, idx] / row_totals[row, 0] * 100
                label = f"{val} " if not percentages else f"{pct:.0f}%"   # or {pct:.0f}% for integer %

                x_center = lefts[idx][row] + widths[idx][row] / 2
                ax.text(
                    x_center,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                    zorder=20
                )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(groups, fontsize=fontsize)
    ax.set_title(plot_title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Symmetric axis
    max_right = np.max(left5 + c5)
    max_left  = np.min(left1)
    lim = max(abs(max_left), abs(max_right))
    ax.set_xlim(-lim, lim)

    # Custom ticks: less / neutral / more
    
    if x_axis_lables:
        ax.set_xticks([-lim, 0, lim])
        ax.set_xticklabels(
            x_axis_lables,
            fontsize=fontsize
        )

    # Legend
    if category_labels is None:
        category_labels = {i: str(i) for i in cats}

    if legend_title:
        ax.legend(title=legend_title)
    
    legend_handles = []
    legend_texts = []
    for i in cats:
        label = category_labels.get(i, "")
        if label:                         # keep only non-empty labels
            legend_handles.append(plt.Rectangle((0,0),1,1,color=colors[i]))
            legend_texts.append(label)

    ax.legend(legend_handles, legend_texts, loc=legend_loc)



    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()

