import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib import transforms


def _wrap_lines(text: str, max_chars_per_line: int = 12) -> str:
    words = text.replace("_", " ").split()
    lines, cur = [], ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars_per_line:
            cur = f"{cur} {w}"
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)


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
    plot_title="",
    show_n=False,  # annotate each row with its number of participants
    correct_categories=None,  # kept for API compatibility (unused)
):
    groups = list(data.keys())
    cats = [1, 2, 3, 4, 5]

    # ---------- scenario grouping: rows named "<scenario> (R)" are paired with "<scenario>" ----------
    scenario_of = []
    is_relevant_row = []
    for g in groups:
        if g.endswith(" (R)"):
            scenario_of.append(g[:-4])
            is_relevant_row.append(True)
        else:
            scenario_of.append(g)
            is_relevant_row.append(False)
    grouped = any(is_relevant_row)

    scenario_order = []
    scenario_to_indices = {}
    for i, s in enumerate(scenario_of):
        if s not in scenario_to_indices:
            scenario_to_indices[s] = []
            scenario_order.append(s)
        scenario_to_indices[s].append(i)

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

    # ---------- layout ----------
    if grouped:
        # pair rows per scenario (close together), scenarios apart (far)
        bar_h = 0.90
        intra_gap = 0.08
        inter_gap = 0.85

        y_pos = np.zeros(len(groups), dtype=float)
        y = 0.0
        for s in scenario_order:
            idxs = scenario_to_indices[s]
            for j, idx in enumerate(idxs):
                y_pos[idx] = y
                if j < len(idxs) - 1:
                    y += 1.0 + intra_gap
            y += 1.0 + inter_gap

        span = float(y_pos.max() - y_pos.min()) if len(y_pos) > 1 else 1.0
        fig_height = max(4.0, 1.8 + 0.35 * span)
    else:
        bar_h = 0.6
        y_pos = np.arange(len(groups), dtype=float)
        fig_height = 4 if len(data) > 5 else 3

    fig, ax = plt.subplots(figsize=(9, fig_height))

    # 0-line behind everything
    ax.axvline(0, color="black", linewidth=1, linestyle="--", zorder=0)

    # Draw bars and labels
    value_fontsize = (max(fontsize, 12) + 2) if grouped else 9

    # If a segment is too small, place its label just left of the segment instead of inside it
    min_width_for_inside = 6.0 if normalize else 2.0
    label_pad = 0.8 if normalize else 0.2

    for idx, cat in enumerate(cats):
        for row in range(len(groups)):
            w = float(widths[idx][row])
            if w <= 0:
                continue

            x_left = float(lefts[idx][row])
            y_center = float(y_pos[row])

            ax.barh(
                y_center,
                w,
                left=x_left,
                height=bar_h,
                color=colors[cat],
                zorder=10,
                edgecolor="none",
                linewidth=0
            )

            # Add absolute + percentage on top: "12 (34%)"
            val = int(raw_counts[row, idx])
            if val > 0:
                pct = raw_counts[row, idx] / row_totals[row, 0] * 100
                label = f"{val} " if not percentages else f"{pct:.0f}%"   # or {pct:.0f}% for integer %

                if w < min_width_for_inside:
                    ax.text(
                        x_left - label_pad,
                        y_center,
                        label,
                        ha="right",
                        va="center",
                        fontsize=value_fontsize,
                        color=text_color,
                        zorder=20
                    )
                else:
                    ax.text(
                        x_left + w / 2,
                        y_center,
                        label,
                        ha="center",
                        va="center",
                        fontsize=value_fontsize,
                        color=text_color,
                        zorder=20
                    )

    # ---------- y labels ----------
    if grouped:
        row_labels = ["⚡" if r else "◻️" for r in is_relevant_row]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(row_labels, fontsize=fontsize)

        # rotated scenario labels to the left of the y-ticks
        blended = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        scenario_label_x = -0.08
        for s in scenario_order:
            idxs = scenario_to_indices[s]
            y_mid = float(np.mean([y_pos[i] for i in idxs]))
            ax.text(
                scenario_label_x,
                y_mid,
                _wrap_lines(s),
                transform=blended,
                rotation=90,
                ha="center",
                va="center",
                fontsize=fontsize,
                linespacing=1.35,
            )
        fig.subplots_adjust(left=0.2)
    else:
        ax.set_yticks(y_pos)
        ax.set_yticklabels(groups, fontsize=fontsize)

    ax.set_title(plot_title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ---------- x axis ----------
    max_right = np.max(left5 + c5)
    max_left = np.min(left1)

    if grouped:
        # asymmetric limits with extra left padding for the outside labels/scenario text
        rng = (max_right - max_left) if (max_right > max_left) else 1.0
        left_pad = 0.08 * rng
        right_pad = 0.005 * rng
        ax.set_xlim(max_left - left_pad, max_right + right_pad)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    else:
        # Symmetric axis (unchanged from before)
        lim = max(abs(max_left), abs(max_right))
        ax.set_xlim(-lim, lim)

        # Custom ticks: less / neutral / more
        if x_axis_lables:
            ax.set_xticks([-lim, 0, lim])
            ax.set_xticklabels(
                x_axis_lables,
                fontsize=fontsize
            )

    if grouped:
        ax.set_ylim(y_pos.min() - (bar_h / 2 + 0.8), y_pos.max() + (bar_h / 2 + 0.8))

    # Legend
    if show_n:
        # number of participants per row, just right of the plot area
        for row in range(len(groups)):
            ax.text(1.01, y_pos[row], f"n={int(row_totals[row, 0])}", transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=fontsize, color="dimgray", clip_on=False)

    if category_labels is None:
        category_labels = {i: str(i) for i in cats}

    legend_handles = []
    legend_texts = []
    for i in cats:
        label = category_labels.get(i, "")
        if label:                         # keep only non-empty labels
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=colors[i]))
            legend_texts.append(label)

    if legend_loc == "below":
        # centre the legend on the visible figure content (y-axis labels through the right end of
        # the axis), measured after layout so the label widths are taken into account
        fig.tight_layout()
        fig.canvas.draw()
        bb = ax.get_tightbbox(fig.canvas.get_renderer())
        (x0, _), (x1, _) = ax.transAxes.inverted().transform([(bb.x0, 0), (bb.x1, 0)])
        legend_x = (x0 + x1) / 2
        # single row underneath the bars so it can never cover bars or their labels
        # leave room for the numeric x tick labels of the ungrouped layout
        legend_y = -0.03 if grouped else -0.16
        ax.legend(legend_handles, legend_texts, loc="upper center", bbox_to_anchor=(legend_x, legend_y),
                  ncol=len(legend_texts), frameon=False, columnspacing=1.2, handlelength=1.2,
                  handletextpad=0.5, title=legend_title if legend_title else None)
    else:
        ax.legend(legend_handles, legend_texts, loc=legend_loc, title=legend_title if legend_title else None)

    plt.tight_layout()
    # bbox_inches="tight" keeps a legend that is wider than the axes from being clipped
    plt.savefig(out_path, dpi=300, bbox_inches="tight" if legend_loc == "below" else None)
    if plt.get_backend().lower() != "agg":
        plt.show()
