import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from tmp.plot_likert import plot_likert_5point


def plot_array_barplot (input_array, container, out_file, x_label, y_label, show_numbers_in_plots=True):
    fig, ax = plt.subplots(figsize=(12, 6))
    # Format ticks as integers
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
    sns.barplot(x=input_array.index, y=input_array.values, ax=ax)

    if show_numbers_in_plots:
        # Add labels on bars
        ax.bar_label(ax.containers[0], padding=2)

    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.tight_layout()
    plt.xticks(rotation=90)

    plt.savefig(out_file[:-4]+'.pdf', dpi=300, bbox_inches='tight')

    
    col = st.columns([1,3,1])
    col[1].pyplot(fig, use_container_width=False)



def plot_countplot(df, container, column_index, out_file, x_label, y_label, y_orientation=True, center_column_width=3, show_numbers_in_plots=True):
    if y_orientation:
        fig, ax = plt.subplots(figsize=(3, 0.35 * len(df)))
        sns.countplot(y=column_index, data=df, ax=ax)
    else:
        fig, ax = plt.subplots(figsize=(3, 2))
        sns.countplot(x=column_index, data=df, ax=ax)

    if show_numbers_in_plots:
        # Add labels on bars
        ax.bar_label(ax.containers[0], padding=2)
    
    ymax = ax.get_ylim()[1]
    ax.set_ylim(top=ymax * 1.2)

    # Set axis labels and title
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    plt.tight_layout()
    plt.savefig(out_file[:-4]+'.pdf', dpi=300, bbox_inches='tight')

    
    col = st.columns([1,center_column_width,1])
    col[1].pyplot(fig, use_container_width=False)


def plot_df_barplot(df, container, column_index, out_file, x_label, y_label, show_numbers_in_plots=True, x_lim_adjustment=False, subplots=False, custom_width=False):
    if custom_width:
        fig, ax = plt.subplots(figsize=(custom_width, 0.35 * len(df)))
    else:
        fig, ax = plt.subplots(figsize=(9, 0.35 * len(df)))
    if subplots:
        sns.barplot(y=column_index, x='Count', hue='Is Relevant', data=df, ax=ax)
    else:
        sns.barplot(y=column_index, x='Count', data=df, ax=ax)

    if show_numbers_in_plots:
        # Add labels on bars
        ax.bar_label(ax.containers[0], padding=2)
        if x_lim_adjustment:
            x0, x1 = ax.get_xlim()
            ax.set_xlim(x0, x1 * x_lim_adjustment)

    # Set axis labels and title
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    plt.tight_layout()
    plt.savefig(out_file[:-4]+'.pdf', dpi=300, bbox_inches='tight')

    
    col = st.columns([1,3,1])
    col[1].pyplot(fig, use_container_width=False)


def plot_countplot_array(df, container, columns, titles, out_file, x_label, y_label, show_numbers_in_plots=True):
    # Set up 1 row, 4 columns of subplots
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)

    # Loop through each column and subplot
    for ax, column, title in zip(axes, columns, titles):
        sns.countplot(x=df[column], order=[1, 2, 3, 4, 5], ax=ax)
        if show_numbers_in_plots:
            ax.bar_label(ax.containers[0], padding=2)
        ax.set_title(title)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)

    plt.tight_layout()
    plt.savefig(out_file[:-4]+'.pdf', dpi=300)
    
    col = st.columns([1,3,1])
    col[1].pyplot(fig, use_container_width=False)


def plot_perceived_importance_likert(out_path):
    importance_data = {
        "... your customers":  {1: 58, 2: 18, 3: 16, 4: 18, 5: 14},
        "... your company":    {1: 54, 2: 20, 3: 20, 4: 21, 5: 16},
        "... your colleagues": {1: 50, 2: 24, 3: 22, 4: 21, 5: 10},
        "... you":             {1: 38, 2: 17, 3: 20, 4: 36, 5: 23},
    }
        

    category_labels = {
        1: "1 - Unimportant",
        2: "2",
        3: "3 - Neutral",
        4: "4",
        5: "5 - Important"
    }

    my_colors = {
        1: "#a227d7",
        2: "#cb59fc",
        3: "#c9c7c7",
        4: "#91bfdb",
        5: "#4575b4"
    }

    plot_likert_5point(
        importance_data,
        out_path=out_path,
        normalize=True,
        category_labels=category_labels,
        category_colors=my_colors,
        fontsize=12,
        percentages=False,
        plot_title="How important is the energy consumption of your software product for ...?"

    )

def plot_scenario_likert(out_path):
    scenario_data = {
        "No correlation": {1: 0, 2: 0, 3: 26, 4: 49, 5: 1},
        "Negative correlation": {1: 3, 2: 18, 3: 20, 4: 32, 5: 10},
        "Moderate positive correlation": {1: 0, 2: 2, 3: 8, 4: 54, 5: 19},
        "Strong positive correlation": {1: 0, 2: 2, 3: 6, 4: 47, 5: 29},
    }

    category_labels = {
        5: "Perfect positive",
        4: "Moderate positive",
        3: "No correlation",
        2: "Moderate negative",
        1: "Perfect negative"
    }

    my_colors = {
        1: "#a227d7",
        2: "#cb59fc",
        3: "#c9c7c7",
        4: "#91bfdb",
        5: "#4575b4"
    }

    plot_likert_5point(
        scenario_data,
        out_path=out_path,
        legend_title="Correlation",
        legend_loc="center left",
        normalize=False,
        category_labels=category_labels,
        category_colors=my_colors,
        fontsize=12,
        percentages=False,
        #plot_title="Responses to 'How important is the energy consumption of your software product for ...'"
    )

def plot_proxy_suitability_likert(out_path):
    proxy_data = {
        "Network utilization": {1: 9,  2: 22, 3: 0,  4: 27, 5: 12},
        "Cloud provider cost": {1: 8,  2: 15, 3: 0,  4: 22, 5: 8},
        "RAM utilization":     {1: 15, 2: 18, 3: 0,  4: 33, 5: 11},
        "Disc utilization":    {1: 6,  2: 20, 3: 0,  4: 36, 5: 11},
        "Electricity bills":   {1: 11, 2: 8,  3: 0,  4: 20, 5: 21},
        "Runtime performance": {1: 2,  2: 6,  3: 0,  4: 40, 5: 27},
        "CPU utilization":     {1: 0,  2: 6,  3: 0,  4: 29, 5: 44},
        "Battery utilization": {1: 1,  2: 3,  3: 0,  4: 14, 5: 45},
    }

    category_labels = {
        5: "Highly suitable",
        4: "Somewhat suitable",
        3: "",
        2: "Somewhat unsuitable",
        1: "Completely unsuitable",
    }

    my_colors = {
        1: "#a227d7",
        2: "#cb59fc",
        3: "#c9c7c7",
        4: "#91bfdb",
        5: "#4575b4"
    }

    plot_likert_5point(
        proxy_data,
        out_path=out_path,
        legend_loc="center left",
        normalize=False,
        category_labels=category_labels,
        category_colors=my_colors,
        fontsize=12,
        percentages=False,
        x_axis_lables=['Unsuitable','','Suitable'],
        #plot_title="Responses to 'How important is the energy consumption of your software product for ...'"
    )

def plot_scenario_likert_relevance(out_path):
    scenario_data = {
    "No correlation": {1: 0, 2: 0, 3: 19, 4: 35, 5: 0},
    "No correlation (R)":       {1: 0, 2: 0, 3: 7, 4: 14, 5: 1},

    "Negative correlation": {1: 1, 2: 12, 3: 14, 4: 22, 5: 10},
    "Negative correlation (R)":       {1: 2, 2: 6, 3: 6, 4: 10, 5: 2},

    "Moderate positive correlation": {1: 0, 2: 2, 3: 5, 4: 37, 5: 12},
    "Moderate positive correlation (R)":       {1: 0, 2: 0, 3: 3, 4: 17, 5: 7},

    "Strong positive correlation": {1: 0, 2: 2, 3: 6, 4: 47, 5: 29},
    "Strong positive correlation (R)":       {1: 0, 2: 0, 3: 1, 4: 16, 5: 10}, 
}

    category_labels = {
        5: "Perfect positive",
        4: "Moderate positive",
        3: "No correlation",
        2: "Moderate negative",
        1: "Perfect negative"
    }

    my_colors = {
        1: "#a227d7",
        2: "#cb59fc",
        3: "#c9c7c7",
        4: "#91bfdb",
        5: "#4575b4"
    }

    plot_likert_5point(
        scenario_data,
        out_path=out_path,
        legend_title="Correlation",
        legend_loc="center left",
        normalize=True,
        category_labels=category_labels,
        category_colors=my_colors,
        fontsize=12,
        percentages=True,
        #plot_title="Responses to 'How important is the energy consumption of your software product for ...'"
    )