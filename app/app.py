import pandas as pd

import streamlit as st
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import json
import itertools
import logging
import random
import concurrent.futures
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, kruskal, rankdata
from scipy.stats import norm as _scipy_norm
for name, l in logging.root.manager.loggerDict.items():
    if "streamlit" in name:
        l.disabled = True

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
# rpy2 warns once when R gets embedded outside the main thread — expected here, since all R calls
# are deliberately funneled through a dedicated worker thread (see _get_r_executor).
warnings.filterwarnings("ignore", message="R is not initialized by the main thread")


# Global Variables
lime_survey_csv = './app/data/results-survey762399.csv'
card_sorting_categories_csv = './app/data/Card-Sorting.csv'



# Figures
import plots as custom_plt


# Streamlit Page Config
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="Untersuchung zu energiebewusster Softwareentwicklung"
)

st.sidebar.title("Research Questions")
page = st.sidebar.radio("", [ 
    "RQ 1: Relevance", 
    "RQ 2: Assessment",
    "RQ 3: Mitigation Strategies",
    "RQ 4: Proxies Old",
    "RQ 4: Proxy Analysis Revisions",
    "RQ 5: Incentives",
    # "Revision: Extended Proxys Evaluation",  # hidden from the menu
    "Personal Background",
    # "Fundamental Disagreements",
    "Data Exploration"], key="page")

st.sidebar.title("Page Config")
show_numbers_in_plots = st.sidebar.checkbox("Show raw numbers in plots", value=True)
show_category_graph = st.sidebar.checkbox("Show category graph", value=True)


# Helper Functions
def add_paragraph(n=7):
    for _ in range(n):
        st.write(" ")


# Controller
def prep_PQ1(df, container):
    container.write("##### PQ1: Wie viele Jahre arbeiten Sie bereits in der Softwareentwicklung?")
    counts = df['PQ1'].value_counts().sort_index()
    custom_plt.plot_array_barplot(counts, container, './app/figures/years_experience.png', 'Years of experience', 'Developers', show_numbers_in_plots)


def prep_PQ2(df, container):
    container.write("##### PQ2: Was ist Ihr Erfahrungslevel?")

    # Restrict to participants who attended past the initial screening (answered RQ4) — the
    # N=134 "Attended" cohort of the Sankey flow, also used for all other Personal Background
    # questions; the 90 "Immediate Stop" participants never reached this question. (The
    # scenario analyses use a further subset: the 65-72 participants per scenario who answered
    # the proxy scenarios.)
    df = df[df['RQ4'].notna()].copy()

    mask = df['PQ2'] == 'Sonstiges'
    df.loc[mask, 'PQ2'] = df.loc[mask].apply(
        lambda row: f"{row['PQ2[other]']}" if pd.notna(row['PQ2[other]']) else row['PQ2'], axis=1
    )
    df.loc[:, 'PQ2'] = df['PQ2'].replace("we don't have a Junior/Senior split, but it would be Senior probably ", "Senior")

    pq2_counts_df = df['PQ2'].value_counts().reset_index()
    pq2_counts_df.columns = ['Category', 'Count']
    pq2_counts_df = pq2_counts_df.sort_values('Count', ascending=False)

    custom_plt.plot_df_barplot(pq2_counts_df, container, 'Category', './app/figures/seniority.png', 'Developers', 'Seniority', x_lim_adjustment=1.05)

    seniority_col = df['PQ2'].apply(_classify_seniority)
    seniority_counts = seniority_col.value_counts()
    # 'Mid' is omitted here: raw 'Mid'/'Mid Level' free-text answers are already normalized to
    # 'Senior' by the `mapping` dict before this classification runs, so no participant is ever
    # actually classified as 'Mid' — keeping it out avoids a permanently-zero row.
    container.write(
        f"**Seniority levels used in the significance analyses** (N={len(df)}): " +
        ", ".join(f"{lvl}={seniority_counts.get(lvl, 0)}" for lvl in ['Junior', 'Senior', 'Lead']) +
        f", unclassified/excluded={int(seniority_col.isna().sum())}"
    )

    pq3_category_mapping = {
        'PQ3[SQ001]': 'Mobile',
        'PQ3[SQ002]': 'IoT',
        'PQ3[SQ003]': 'Webentwicklung',
        'PQ3[SQ004]': 'Infrastruktur',
        'PQ3[SQ005]': 'Enterprise',
    }
    pq3_col = df.apply(
        decode_multiple_choice_row, axis=1,
        category_mapping=pq3_category_mapping, other_cols=['PQ3[other]']
    )
    membership_col = pq3_col.apply(_domain_memberships)
    domain_selection_counts = {dom: int(membership_col.apply(lambda doms, d=dom: d in doms).sum())
                               for dom in _ALL_DOMAINS}
    n_unclassified = int(membership_col.apply(len).eq(0).sum())
    container.write(
        f"**Domain selections (descriptive, N={len(df)} participants)** — a participant naming "
        "multiple domains counts toward each, so totals can exceed N: " +
        ", ".join(f"{dom}={cnt}" for dom, cnt in domain_selection_counts.items()) +
        f", unclassified={n_unclassified}"
        f", total selections={sum(domain_selection_counts.values())}"
    )

    energy_group_col = pq3_col.apply(_energy_domain_group)
    energy_group_counts = energy_group_col.value_counts()
    container.write(
        f"**Domain groups used in the significance analysis** (N={len(df)}) — "
        f"Energy-aware (IoT/Mobile/Embedded)={energy_group_counts.get('Energy-aware (IoT/Mobile/Embedded)', 0)}, "
        f"Other domains={energy_group_counts.get('Other domains', 0)}; "
        f"excluded (named both groups)={energy_group_counts.get('Both', 0)}, "
        f"excluded (no PQ3 answer)={int(energy_group_col.isna().sum())}"
    )

def prep_PQ3(df, container):
    container.write("##### PQ3: In welchem Bereich entwickeln Sie Software?")
    # 'PQ3[SQ001]', 'PQ3[SQ002]', 'PQ3[SQ003]', 'PQ3[SQ004]', 'PQ3[SQ005]', 'PQ3[other]'
    # Mobile, IoT, Webentwicklung, Infrastruktur, Enterprise, Sonstiges:

    category_mapping = {
        'PQ3[SQ001]': 'Mobile',
        'PQ3[SQ002]': 'IoT',
        'PQ3[SQ003]': 'Webentwicklung',
        'PQ3[SQ004]': 'Infrastruktur',
        'PQ3[SQ005]': 'Enterprise'
    }

    counts = {}
    for col, name in category_mapping.items():
        counts[name] = (df[col] == 'Ja').sum()

    other_counts = df['PQ3[other]'].dropna().value_counts()

    # First, make DataFrame of main counts
    main_counts_df = pd.DataFrame(list(counts.items()), columns=['Category', 'Count'])

    # Now create the detailed Sonstiges DataFrame
    detailed_other_df = other_counts.reset_index()
    detailed_other_df.columns = ['Category', 'Count']

    full_counts_df = pd.concat([main_counts_df, detailed_other_df], ignore_index=True)
    full_counts_df_sorted = full_counts_df.sort_values('Count', ascending=False)

    custom_plt.plot_df_barplot(full_counts_df_sorted, container, 'Category', './app/figures/professional_background.png', 'Developers', 'Professional background')

def prep_PQ4(df, container):
    container.write("##### PQ4: An welchen Teilen von Softwareprodukten arbeiten Sie hauptsächlich?")
    # 'PQ4[SQ001]', 'PQ4[SQ002]', 'PQ4[SQ003]', 'PQ4[SQ004]', 'PQ4[SQ005]', 'PQ4[SQ006]', 'PQ4[other]'
    # 'Frontend', 'Backend', 'Full-Stack', 'DevOps', 'Data/Database', 'Design/Architecture', 'PQ4[other]'

    pq4_category_mapping = {
        'PQ4[SQ001]': 'Frontend',
        'PQ4[SQ002]': 'Backend',
        'PQ4[SQ003]': 'Full-Stack',
        'PQ4[SQ004]': 'DevOps',
        'PQ4[SQ005]': 'Data/Database',
        'PQ4[SQ006]': 'Design/Architecture'
    }

    counts = {}
    for col, name in pq4_category_mapping.items():
        counts[name] = (df[col] == 'Ja').sum()

    pq4_other_counts = df['PQ4[other]'].dropna().value_counts()
    pq4_main_counts_df = pd.DataFrame(list(counts.items()), columns=['Category', 'Count'])
    pq4_detailed_other_df = pq4_other_counts.reset_index()
    pq4_detailed_other_df.columns = ['Category', 'Count']
    # Merge main and Sonstiges dfs
    pq4_full_counts_df = pd.concat([pq4_main_counts_df, pq4_detailed_other_df], ignore_index=True)

    # Normalize category names
    pq4_full_counts_df.loc[:, 'Category'] = pq4_full_counts_df['Category'].str.strip().str.title()
    pq4_counts_deduplicated = pq4_full_counts_df.groupby('Category', as_index=False)['Count'].sum()
    pq4_full_counts_sorted = pq4_counts_deduplicated.sort_values('Count', ascending=False)

    custom_plt.plot_df_barplot(pq4_full_counts_sorted, container, 'Category', './app/figures/software_product_parts.png', 'Developers', 'Kategorie')


def prep_RQ1(df, container):
    container.write("##### RQ1: Wie wichtig ist der Energieverbrauch der Software, die Sie entwickeln, für... ?")
    container.write("1 - unwichtig ... 5 - wichtig")
    mapping = {
        '1 - unwichtig': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5 - wichtig': 5
    }

    # Define the questions and their titles
    titles = ['... you?', '... your colleagues?', '... your company?', '... your customers?']
    columns = ['RQ1[SQ001]', 'RQ1[SQ002]', 'RQ1[SQ003]', 'RQ1[SQ004]']

    df_mapped = df[columns].map(
        lambda x: mapping.get(str(x).strip()) if pd.notna(x) else None
    )
    df_mapped_int = df_mapped.map(lambda x: int(x) if pd.notna(x) else pd.NA)

    custom_plt.plot_countplot_array(df_mapped_int, container, columns, titles, './app/figures/importance_distributions_multiple.png', 'Importance', 'Developers')

def prep_RQ2(df, container):
    container.write("##### RQ2: Aus Ihrer Sicht, lohnt sich der Aufwand, Energieverbrauch durch Optimierung von Software zu reduzieren? Bitte erläutern Sie.")

    question_column = 'RQ2'
    filtered_df = df[pd.notna(df[question_column])]
    generate_open_answer_expander(filtered_df, container, question_column)

def prep_RQ4(df, container):
    container.write("##### RQ4: Ist der Energieverbrauch der Software, die Sie entwickeln, in Ihren Projekten ein relevantes Thema, das Ihre Entscheidungen und Ihr Handeln beeinflusst?")
    container.write("Nein, der Energieverbrauch ist nicht relevantfür in unseren Projekten, obwohl wir uns auf andere Eigenschaften (wie Laufzeit oder Cloud-Kosten) konzentrieren, die wiederum den Energieverbrauch beeinflussen können. ")
    container.write("Ja, der Energieverbrauch ist relevant in unseren Projekten und wir bewerten den Energieverbrauch unserer Software (z. B. über Energiemessungen, Laufzeitleistung, CPU-Auslastung, Kundenfeedback, Bauchgefühl oder anderes).")

    df = df.copy()
    rq4_mapping = {
        "Nein, der Energieverbrauch ist nicht relevantfür in unseren Projekten, obwohl wir uns auf andere Eigenschaften (wie Laufzeit oder Cloud-Kosten) konzentrieren, die wiederum den Energieverbrauch beeinflussen können.": "Not relevant",
        "Ja, der Energieverbrauch ist relevant in unseren Projekten und wir bewerten den Energieverbrauch unserer Software (z. B. über Energiemessungen, Laufzeitleistung, CPU-Auslastung, Kundenfeedback, Bauchgefühl oder anderes).": "Is relevant",
    }
    df.loc[:, 'RQ4'] = df['RQ4'].map(lambda x: rq4_mapping.get(x, x))
    filtered_df = df[pd.notna(df['RQ4'])]

    custom_plt.plot_countplot(filtered_df, container, 'RQ4', './app/figures/energy_consumption_relevant_topic.png', '', 'Developers', False, 1)

def prep_Filter_N2(df, container):

    container.write("##### FilterN2: Do you focus on other factors (such as cloud cost, runtime, CPU utilization, battery  usage, selection of efficient HW, etc.) that might influence energy consumption?")
    filtered_df = df[pd.notna(df['FilterN2'])]

    custom_plt.plot_countplot(filtered_df, container, 'FilterN2', './app/figures/focus_on_proxy_values.png', '', 'Developers', False, 1)
    return

def prep_EPT0(df, container):
    container.write("##### EPT0: Auf welche Entscheidungen hat der Energieverbrauch, der Software die Sie entwickeln, in Ihrem Arbeitsalltag einen Einfluss?")
    # EPT0[SQ001]	EPT0[SQ002]	EPT0[SQ003]	EPT0[SQ004]	EPT0[other]

    category_mapping = {
        'EPT0[SQ001]': 'Hardware components',
        'EPT0[SQ002]': 'Frameworks and libraries',
        'EPT0[SQ003]': 'Design and construction',
        'EPT0[SQ004]': 'Source-code level'
    }
    # Decisions on specific hardware components (e.g., ARM64 vs. X64, CPUs, GPUs, servers)
    # Decisions on specific frameworks/libraries (e.g., databases, APIs, Bluetooth Low Energy)
    # Decisions on Software design and construction (e.g., choosing architectures and programming languages)
    # Decisions on source-code level (e.g., selecting energy efficient data structures, coding best practices)

    counts = {}
    for col, name in category_mapping.items():
        counts[name] = (df[col] == 'Ja').sum()

    other_counts = df['EPT0[other]'].dropna().value_counts()

    # First, make DataFrame of main counts
    main_counts_df = pd.DataFrame(list(counts.items()), columns=['Category', 'Count'])

    # Now create the detailed Sonstiges DataFrame
    detailed_other_df = other_counts.reset_index()
    detailed_other_df.columns = ['Category', 'Count']

    full_counts_df = pd.concat([main_counts_df, detailed_other_df], ignore_index=True)

    merged_df = (
        full_counts_df
        .groupby('Category', as_index=False)['Count']
        .sum()
    )

    full_counts_df_sorted = merged_df.sort_values('Count', ascending=False)

    custom_plt.plot_df_barplot(full_counts_df_sorted, container, 'Category', './app/figures/energy_influencing_decisions.png', '', '')

def prep_EPT1(df, container):
    container.write("##### ETP1: Wie oft treffen Sie Entscheidungen auf der Grundlage des Energieverbrauchs?")

    filtered_df = df[pd.notna(df['EPT1'])]

    custom_plt.plot_countplot(filtered_df, container, 'EPT1', './app/figures/energy_consumption_decision_frquency.png', 'Frequency', 'Count', False, 1)


def prep_Filter_Y2(df, container):
    container.write("##### Filter Y2: Wie ermitteln Sie den Energieverbrauch Ihrer Software?")
    # FilterY2[direct1]	FilterY2[direct2]	FilterY2[indirect1]	FilterY2[indirect2]	FilterY2[indirect3]	FilterY2[indirect4]	FilterY2[indirect5]	FilterY2[indirect6]	FilterY2[indirect7]	FilterY2[indirect8]	FilterY2[other1]	FilterY2[other2]	FilterY2[other]

    category_mapping = {
        'FilterY2[direct1]': 'Entire system (physical measurement)',
        'FilterY2[direct2]': 'RAPL (physical measurement)',
        'FilterY2[indirect1]': 'Execution time',
        'FilterY2[indirect2]': 'CPU usage',
        'FilterY2[indirect3]': 'RAM usage',
        'FilterY2[indirect4]': 'Disc (I/O) usage',
        'FilterY2[indirect5]': 'Network usage',
        'FilterY2[indirect6]': 'Battery usage',
        'FilterY2[indirect7]': 'Cloud provider cost',
        'FilterY2[indirect8]': 'Electricity bills',
        'FilterY2[other1]': 'Based on customer feedback',
        'FilterY2[other2]': 'Based on feeling or intuition'
    }

    counts = {}
    for col, name in category_mapping.items():
        counts[name] = (df[col] == 'Ja').sum()

    other_counts = df['FilterY2[other]'].dropna().value_counts()

    # First, make DataFrame of main counts
    main_counts_df = pd.DataFrame(list(counts.items()), columns=['Category', 'Count'])

    # Now create the detailed Sonstiges DataFrame
    detailed_other_df = other_counts.reset_index()
    detailed_other_df.columns = ['Category', 'Count']

    full_counts_df = pd.concat([main_counts_df, detailed_other_df], ignore_index=True)
    full_counts_df_sorted = full_counts_df.sort_values('Count', ascending=False)

    custom_plt.plot_df_barplot(full_counts_df_sorted, container, 'Category', './app/figures/energy_consumption_assessment_strategies.png', 'Developers', '')

def prep_AE2(df, container):
    # AE2[SQ001]	AE2[SQ002]	AE2[SQ003]	AE2[SQ005]	AE2[SQ004]	AE2[SQ006]	AE2[other]
    container.write("##### AE2: Wann wird der Energieverbrauch Ihrer Software ermittelt?")

    category_mapping = {
        'AE2[SQ001]': 'After each commit',
        'AE2[SQ002]': 'After each feature completion',
        'AE2[SQ003]': 'After each release/deployment',
        'AE2[SQ005]': 'Irregularly (based on intuition)',
        'AE2[SQ004]': 'Irregularly (based on feedback or request)',
        'AE2[SQ006]': 'Never'
    }

    counts = {}
    for col, name in category_mapping.items():
        counts[name] = (df[col] == 'Ja').sum()

    other_counts = df['AE2[other]'].dropna().value_counts()

    # First, make DataFrame of main counts
    main_counts_df = pd.DataFrame(list(counts.items()), columns=['Category', 'Count'])

    # Now create the detailed Sonstiges DataFrame
    detailed_other_df = other_counts.reset_index()
    detailed_other_df.columns = ['Category', 'Count']

    full_counts_df = pd.concat([main_counts_df, detailed_other_df], ignore_index=True)
    full_counts_df_sorted = full_counts_df.sort_values('Count', ascending=False)

    custom_plt.plot_df_barplot(full_counts_df_sorted, container, 'Category', './app/figures/energy_consumption_assessment_frequency.png', 'Developers', '', x_lim_adjustment=1.07)

def prep_AE3(df, container):

    container.write("##### AE3: Auf welcher Granularitätsstufe erfassen Sie den Energieverbrauch?")

    category_mapping = {
        'AE3[SQ001]': 'Code level',
        'AE3[SQ002]': 'Feature level',
        'AE3[SQ003]': 'Module level',
        'AE3[SQ004]': 'System level',
        'AE3[SQ005]': 'Development Process'
    }

    counts = {}
    for col, name in category_mapping.items():
        counts[name] = (df[col] == 'Ja').sum()

    other_counts = df['AE3[other]'].dropna().value_counts()

    # First, make DataFrame of main counts
    main_counts_df = pd.DataFrame(list(counts.items()), columns=['Category', 'Count'])

    # Now create the detailed Sonstiges DataFrame
    detailed_other_df = other_counts.reset_index()
    detailed_other_df.columns = ['Category', 'Count']

    full_counts_df = pd.concat([main_counts_df, detailed_other_df], ignore_index=True)
    full_counts_df_sorted = full_counts_df.sort_values('Count', ascending=False)

    custom_plt.plot_df_barplot(full_counts_df_sorted, container, 'Category', './app/figures/energy_consumption_assessment_granularity.png', 'Developers', '', x_lim_adjustment=1.05, custom_width=5)

def prep_AE4(df, container):
    container.write("##### AE4: Ist das Erfassen des Energieverbrauchs automatisiert (z.B. innerhalb einer CI/CD-Pipeline)?")

    # AE4 is only shown to participants who found energy consumption relevant (RQ4). Among
    # those, a few completed the whole survey (submitdate set) but left this non-mandatory
    # question blank — the same "reached it, gave no answer" case as the literal 'Keine
    # Antwort' choice, just exported as NaN instead of the pseudo-choice text. Treat both as
    # 'No response' instead of silently dropping the blank ones.
    is_relevant = df['RQ4'].astype(str).str.startswith('Ja,')
    reached_ae4 = df['AE4'].notna() | (is_relevant & df['submitdate'].notna())
    filtered_df = df[reached_ae4].copy()
    filtered_df.loc[:, 'AE4'] = filtered_df['AE4'].fillna('Keine Antwort')

    filtered_df.loc[:, 'AE4'] = filtered_df['AE4'].replace("Ja", "Yes")
    filtered_df.loc[:, 'AE4'] = filtered_df['AE4'].replace("Nein", "No")
    filtered_df.loc[:, 'AE4'] = filtered_df['AE4'].replace("Keine Antwort", "No response")

    custom_plt.plot_countplot(filtered_df, container, 'AE4', './app/figures/energy_consumption_assessment_automated.png', 'Automation', 'Developers', False, 1)

def prep_ECP0(df, container):
    container.write("##### ECP0: Aus welchen Gründen wird der Energieverbrauch Ihrer Software nicht berücksichtigt?")
    # ECP0[SQ001]	ECP0[SQ002]	ECP0[SQ003]	ECP0[SQ004]	ECP0[SQ005]	ECP0[SQ006]	ECP0[SQ007]	ECP0[SQ009]	ECP0[other]

    category_mapping = {
        'ECP0[SQ001]': 'Management is not interested',
        'ECP0[SQ002]': 'Customers are not interested',
        'ECP0[SQ003]': 'Too expensive',
        'ECP0[SQ004]': 'Not enough time',
        'ECP0[SQ005]': 'Too complicated to implement',
        'ECP0[SQ006]': 'Lack of expertise or experience',
        'ECP0[SQ007]': 'Not applicable in our use case',
        'ECP0[SQ009]': 'Not relevant for our use case'
    }

    counts = {}
    for col, name in category_mapping.items():
        counts[name] = (df[col] == 'Ja').sum()

    other_counts = df['ECP0[other]'].dropna().value_counts()

    # First, make DataFrame of main counts
    main_counts_df = pd.DataFrame(list(counts.items()), columns=['Category', 'Count'])

    # Now create the detailed Sonstiges DataFrame
    detailed_other_df = other_counts.reset_index()
    detailed_other_df.columns = ['Category', 'Count']

    full_counts_df = pd.concat([main_counts_df, detailed_other_df], ignore_index=True)
    full_counts_df = full_counts_df.groupby("Category", as_index=False, observed=True)["Count"].sum()
    full_counts_df_sorted = full_counts_df.sort_values('Count', ascending=False)

    custom_plt.plot_df_barplot(full_counts_df_sorted, container, 'Category', './app/figures/reasons_for_no_energy_awareness.png', 'Developers', '')

def prep_EnergyProxy01_02(df, container):
    row1 = st.columns(2)
    row2 = st.columns(2)

    row1[0].write("##### EnergyProxy01: Welche Gründe gibt es für Sie und Ihr Unternehmen, den Energieverbrauch nicht direkt zu messen?")

    question_column = 'EnergyProxy01'
    filtered_df = df[pd.notna(df[question_column])]
    generate_open_answer_expander(filtered_df, row2[0], question_column)


    row1[1].write("### EnergyProxy02: Basierend auf Ihren bisherigen Antworten: Sie beurteilen den Energieverbrauch basierend auf direkten Messungen als auch durch indirekte Messungen. Warum nutzen Sie neben direkten Energiemessungen auch indirekte Messungen?")

    question_column = 'EnergyProxy02'
    filtered_df = df[pd.notna(df[question_column])]
    generate_open_answer_expander(filtered_df, row2[1], question_column)


def prep_ProxyMerge01(df, container):
    container.write(
        "##### ProxyMerge01: Glauben Sie, dass es einen Zusammenhang zwischen Laufzeit (Performance) und Energieverbrauch gibt? "
        "Bitte erläutern Sie Ihre Ansichten, insbesondere Ihre Erwartungen, inwieweit Energieverbrauch und Leistung voneinander abhängig sind oder nicht. "
    )
    question_column = 'ProxyMerge01'
    filtered_df = df[pd.notna(df[question_column])]
    generate_open_answer_expander(filtered_df, container, question_column)

def prep_EnergyProxy03(df, container):
    container.write(
        "##### EnergyProxy03: Wie geeignet sind Ihrer Meinung nach die folgenden Metriken, als Indikator für den Energieverbrauch von Software?"
    )
    # container.write("1 - unwichtig ... 5 - wichtig")
    ordering = ['Sehr gut geeignet', 'Teilweise geeignet', 'Teilweise ungeeignet', 'Komplett ungeeignet', 'Nicht beantwortbar']

    # Define the questions and their titles
    titles = ['Laufzeit', 'CPU Auslastung', 'RAM Auslastung', 'Disk (I/O) Auslastung', 'Netzwerk Auslastung', 'Batterie Auslastung', 'Cloud Provider Kosten', 'Stromrechnung']
    columns = ['EnergyProxy03[SQ001]', 'EnergyProxy03[SQ002]', 'EnergyProxy03[SQ003]', 'EnergyProxy03[SQ004]', 'EnergyProxy03[SQ005]', 'EnergyProxy03[SQ006]', 'EnergyProxy03[SQ007]', 'EnergyProxy03[SQ008]']

    df_mapped = df[columns]
    # Set up 1 row, 4 columns of subplots
    fig, axes = plt.subplots(1, 8, figsize=(20, 5), sharey=True)

    # Loop through each column and subplot
    for ax, column, title in zip(axes, columns, titles):
        sns.countplot(x=df_mapped[column], ax=ax, order=ordering)
        if show_numbers_in_plots:
            ax.bar_label(ax.containers[0])
        ax.set_title(title)
        ax.set_xlabel('x_label')
        ax.set_ylabel('y_label')
        ax.tick_params(axis='x', labelrotation=90)

    plt.tight_layout()
    plt.savefig('./app/figures/energy_proxy_metric_suitability.png', dpi=300)
    container.pyplot(fig, use_container_width=False)

def _prep_scenario_strategy2(df, container, scenario, answer_names, out_png):
    """Bar plot of how often each answer was given in `scenario`, on the Strategy 2 sample (one
    resolved answer per participant, participants with contradicting answers excluded) so it is
    consistent with the rest of the reported results. `answer_names` maps the survey answer string
    to the label shown in the plot."""
    tmp_df, _, _, correct_answers = _prepare_correlation_df(df)
    correct, scale = correct_answers[scenario]
    resolved = _strategy_resolve_noncontradicting(tmp_df, scenario, correct, scale)

    answer_counts = resolved[scenario].value_counts()
    counts = {name: int(answer_counts.get(answer, 0)) for answer, name in answer_names.items()}
    container.write(f"Strategy 2 sample: N={len(resolved)} participants")

    main_counts_df = pd.DataFrame(list(counts.items()), columns=['Category', 'Count'])
    custom_plt.plot_df_barplot(main_counts_df, container, 'Category', out_png, 'Count', 'Korrelation')


_SCENARIO_ANSWER_NAMES_5PT = {
    'perfekte positive Korrelation': 'perfekte positive',
    'moderate positive Korrelation': 'moderate positive Korrelation',
    'moderate negative Korrelation': 'moderate negative Korrelation',
    'perfekte negative Korrelation': 'perfekte negative Korrelation',
    'keine Korrelation': 'keine Korrelation',
}
_SCENARIO_ANSWER_NAMES_3PT = {
    'perfekte Korrelation': 'perfekte Korrelation',
    'moderate Korrelation': 'moderate Korrelation',
    'keine Korrelation': 'keine Korrelation',
}


def prep_Scenario1(df, container):
    container.write(
        "##### Scenario1: Wie geeignet sind Ihrer Meinung nach die folgenden Metriken, als Indikator für den Energieverbrauch von Software?"
    )
    _prep_scenario_strategy2(df, container, 'Scenario1', _SCENARIO_ANSWER_NAMES_5PT,
                             './app/figures/energy_correlation_scenario1.png')


def prep_Scenario2(df, container):
    container.write(
        "##### Scenario2: Wie geeignet sind Ihrer Meinung nach die folgenden Metriken, als Indikator für den Energieverbrauch von Software?"
    )
    _prep_scenario_strategy2(df, container, 'Scenario2', _SCENARIO_ANSWER_NAMES_5PT,
                             './app/figures/energy_correlation_scenario2.png')


def prep_Scenario3(df, container):
    container.write(
        "##### Scenario3: Wie geeignet sind Ihrer Meinung nach die folgenden Metriken, als Indikator für den Energieverbrauch von Software?"
    )
    _prep_scenario_strategy2(df, container, 'Scenario3', _SCENARIO_ANSWER_NAMES_3PT,
                             './app/figures/energy_correlation_scenario3.png')


def prep_Scenario4(df, container):
    container.write(
        "##### Scenario4: Wie geeignet sind Ihrer Meinung nach die folgenden Metriken, als Indikator für den Energieverbrauch von Software?"
    )
    _prep_scenario_strategy2(df, container, 'Scenario4', _SCENARIO_ANSWER_NAMES_5PT,
                             './app/figures/energy_correlation_scenario4.png')


def decode_single_choice_with_other(row, choice_col, other_col, replacements=None):
    value = row.get(choice_col)
    if value == 'Sonstiges':
        other_value = row.get(other_col)
        if pd.notna(other_value) and str(other_value).strip():
            value = str(other_value).strip()

    if replacements:
        value = replacements.get(value, value)

    return value if pd.notna(value) and str(value).strip() else pd.NA


def decode_multiple_choice_row(row, category_mapping, other_cols=None):
    selected_categories = [
        name
        for col, name in category_mapping.items()
        if row.get(col) == 'Ja'
    ]

    for col in other_cols or []:
        other_value = row.get(col)
        if pd.notna(other_value) and str(other_value).strip():
            selected_categories.append(str(other_value).strip())

    return ', '.join(selected_categories) if selected_categories else pd.NA


def _sig_label(p):
    if p < 0.05:
        return "significant (p < 0.05)"
    return "n.s."


def _describe_group(series, label):
    med = series.median()
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    return f"**{label}** — n={len(series)}, Median={med:.2f}, IQR=[{q1:.2f}, {q3:.2f}]"


def _dunn_bonferroni(group_arrays, labels):
    all_data = np.concatenate(group_arrays)
    n = len(all_data)
    ranks = rankdata(all_data)
    split_ranks, idx = [], 0
    for g in group_arrays:
        split_ranks.append(ranks[idx:idx + len(g)])
        idx += len(g)

    pairs = []
    for i in range(len(group_arrays)):
        for j in range(i + 1, len(group_arrays)):
            ni, nj = len(group_arrays[i]), len(group_arrays[j])
            se = np.sqrt((n * (n + 1) / 12) * (1 / ni + 1 / nj))
            z = abs(split_ranks[i].mean() - split_ranks[j].mean()) / se
            p_raw = 2 * (1 - _scipy_norm.cdf(z))
            pairs.append((labels[i], labels[j], z, p_raw))

    m = len(pairs)
    return [(a, b, z, min(p * m, 1.0)) for a, b, z, p in pairs]


def _kruskal_eta2(H, k, n):
    return (H - k + 1) / (n - k) if (n - k) > 0 else float('nan')


def _compute_expert_vs_nonexpert(tmp_df, metric_col):
    tmp_df = tmp_df.copy()
    tmp_df = tmp_df.assign(**{metric_col: pd.to_numeric(tmp_df[metric_col], errors='coerce')})
    expert = tmp_df[tmp_df['RQ4'] == 'Relevant'][metric_col].dropna()
    non_expert = tmp_df[tmp_df['RQ4'] == 'Not relevant'][metric_col].dropna()

    result = {'expert': expert, 'non_expert': non_expert, 'p': None, 'U': None, 'r_rb': None}
    if len(expert) == 0 or len(non_expert) == 0:
        return result

    U, p = mannwhitneyu(expert, non_expert, alternative='two-sided')
    n1, n2 = len(expert), len(non_expert)
    result.update({'p': p, 'U': U, 'r_rb': 2 * U / (n1 * n2) - 1})
    return result


def _render_expert_vs_nonexpert(result, container, metric_col, hide_plot=False):
    container.write("#### Analysis 1: Expert vs. Non-Expert (RQ4)")
    expert, non_expert = result['expert'], result['non_expert']
    container.write(_describe_group(expert, 'Expert (Relevant)'))
    container.write(_describe_group(non_expert, 'Non-Expert (Not relevant)'))

    if result['p'] is None:
        container.write("Insufficient data for test.")
        return

    container.write(
        f"Mann-Whitney U = {result['U']:.1f}, p = {result['p']:.4f} ({_sig_label(result['p'])}), "
        f"rank-biserial r = {result['r_rb']:.3f}"
    )

    plot_df = pd.concat([
        expert.rename(metric_col).to_frame().assign(Group='Expert (Relevant)'),
        non_expert.rename(metric_col).to_frame().assign(Group='Non-Expert (Not relevant)'),
    ])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.stripplot(data=plot_df, x='Group', y=metric_col, jitter=True, alpha=0.6, ax=ax)
    sns.pointplot(data=plot_df, x='Group', y=metric_col, estimator=np.median,
                  errorbar=None, markers='D', color='black', ax=ax)
    ax.set_ylabel(metric_col)
    ax.set_title('Expert vs. Non-Expert')
    plot_container = container.expander("Show plot") if hide_plot else container
    col = plot_container.columns([2,3,2])
    col[1].pyplot(fig, use_container_width=False)
    plt.close(fig)


def _eval_expert_vs_nonexpert(tmp_df, container, metric_col='Mean Distance'):
    result = _compute_expert_vs_nonexpert(tmp_df, metric_col)
    _render_expert_vs_nonexpert(result, container, metric_col)
    return result['p']


def _compute_years_experience(tmp_df, metric_col):
    years = pd.to_numeric(tmp_df['PQ1'], errors='coerce')
    valid = pd.DataFrame({'years': years, 'dist': pd.to_numeric(tmp_df[metric_col], errors='coerce')}).dropna()

    result = {'valid': valid, 'p': None, 'rho': None}
    if len(valid) < 3:
        return result

    rho, p = spearmanr(valid['years'], valid['dist'])
    result.update({'rho': rho, 'p': p})
    return result


def _render_years_experience(result, container, metric_col, hide_plot=False):
    container.write("#### Analysis 2: Years of Experience (PQ1)")
    valid = result['valid']

    if result['p'] is None:
        container.write("Insufficient data for correlation.")
        return

    container.write(_describe_group(valid['years'], 'Years of experience'))
    container.write(
        f"Spearman ρ = {result['rho']:.3f}, p = {result['p']:.4f} ({_sig_label(result['p'])}), n = {len(valid)}"
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.scatterplot(data=valid, x='years', y='dist', alpha=0.7, ax=ax)
    sns.regplot(data=valid, x='years', y='dist', scatter=False, ci=95,
                line_kws={'color': 'red', 'linewidth': 1.5}, ax=ax)
    ax.set_xlabel('Years of experience')
    ax.set_ylabel(metric_col)
    ax.set_title(f'Years of Experience vs. {metric_col} (ρ = {result["rho"]:.2f})')
    plot_container = container.expander("Show plot") if hide_plot else container
    col = plot_container.columns([2,3,2])
    col[1].pyplot(fig, use_container_width=False)
    plt.close(fig)


def _eval_years_experience(tmp_df, container, metric_col='Mean Distance'):
    result = _compute_years_experience(tmp_df, metric_col)
    _render_years_experience(result, container, metric_col)
    return result['p']


_SENIORITY_ORDER = ['Junior', 'Mid', 'Senior', 'Lead']


def _classify_seniority(val):
    """Seniority level for a participant's (decoded) PQ2 answer, via keyword match. 'principal'
    is deliberately not treated as a 'lead' synonym (only a single participant answered
    'Principal' free-text — too small to either stand alone or safely fold into Lead), so it
    falls through to unclassified (pd.NA), same as 'Sole'/'Other'/empty answers."""
    v = str(val).lower()
    if 'junior' in v:
        return 'Junior'
    if 'mid' in v or 'middle' in v or 'mittel' in v:
        return 'Mid'
    if 'senior' in v:
        return 'Senior'
    if 'lead' in v or 'staff' in v:
        return 'Lead'
    return pd.NA


def _compute_seniority(tmp_df, metric_col):
    tmp_df = tmp_df.copy()
    tmp_df = tmp_df.assign(**{metric_col: pd.to_numeric(tmp_df[metric_col], errors='coerce')})

    order = _SENIORITY_ORDER
    seniority = tmp_df['PQ2'].apply(_classify_seniority)
    groups = {lvl: tmp_df[seniority == lvl][metric_col].dropna()
              for lvl in order if (seniority == lvl).any()}

    result = {
        'groups': groups, 'p': None, 'H': None, 'eta2': None, 'k': None, 'n': None,
        'dunn_pairs': [], 'targeted': None,
    }
    if len(groups) < 2:
        return result

    group_vals = [g.values for g in groups.values()]
    group_labels = list(groups.keys())
    n = sum(len(g) for g in group_vals)
    k = len(group_vals)

    H, p = kruskal(*group_vals)
    eta2 = _kruskal_eta2(H, k, n)
    result.update({'p': p, 'H': H, 'eta2': eta2, 'k': k, 'n': n})

    if p < 0.05:
        dunn_pairs = _dunn_bonferroni(group_vals, group_labels)
        result['dunn_pairs'] = dunn_pairs
        dunn_any_sig = any(p_adj < 0.05 for _, _, _, p_adj in dunn_pairs)
        if not dunn_any_sig and 'Lead' in groups:
            non_lead = pd.concat([g for lvl, g in groups.items() if lvl != 'Lead'])
            lead = groups['Lead']
            if len(non_lead) > 0 and len(lead) > 0:
                U_c, p_c = mannwhitneyu(non_lead, lead, alternative='two-sided')
                n1, n2 = len(non_lead), len(lead)
                r_rb = 2 * U_c / (n1 * n2) - 1
                result['targeted'] = {
                    'non_lead': non_lead, 'lead': lead, 'U': U_c, 'p': p_c,
                    'r_rb': r_rb, 'n1': n1, 'n2': n2,
                }

    return result


def _render_seniority(result, container, metric_col, hide_plot=False):
    container.write("#### Analysis 3: Seniority (PQ2)")
    groups = result['groups']
    for label, g in groups.items():
        container.write(_describe_group(g, label))

    if result['p'] is None:
        container.write("Insufficient groups for test.")
        return

    container.write(
        f"Kruskal-Wallis H = {result['H']:.3f}, p = {result['p']:.4f} ({_sig_label(result['p'])}), "
        f"η² = {result['eta2']:.3f}, k = {result['k']}, n = {result['n']}"
    )

    if result['p'] < 0.05:
        container.write("Dunn's post-hoc test (Bonferroni corrected):")
        for a, b, z, p_adj in result['dunn_pairs']:
            container.write(f"  {a} vs {b}: z = {z:.3f}, p_adj = {p_adj:.4f} ({_sig_label(p_adj)})")

        if result['targeted'] is not None:
            t = result['targeted']
            container.write(
                "Dunn's all-pairs found no significant pairs. "
                "Targeted contrast: Junior + Mid + Senior (non-Lead) vs. Lead"
            )
            container.write(_describe_group(t['non_lead'], 'Non-Lead (Junior + Mid + Senior)'))
            container.write(_describe_group(t['lead'], 'Lead'))
            container.write(
                f"Mann-Whitney U = {t['U']:.1f}, p = {t['p']:.4f} ({_sig_label(t['p'])}), "
                f"rank-biserial r = {t['r_rb']:.3f}, n₁={t['n1']}, n₂={t['n2']}"
            )

    present_order = [lvl for lvl in ['Junior', 'Mid', 'Senior', 'Lead'] if lvl in groups]
    plot_df = pd.concat([
        g.rename(metric_col).to_frame().assign(Seniority=lvl)
        for lvl, g in groups.items()
    ])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.stripplot(data=plot_df, x='Seniority', y=metric_col,
                  order=present_order, jitter=True, alpha=0.6, ax=ax)
    sns.pointplot(data=plot_df, x='Seniority', y=metric_col,
                  order=present_order, estimator=np.median,
                  errorbar=None, markers='D', color='black', ax=ax)
    ax.set_ylabel(metric_col)
    ax.set_title(f'Seniority vs. {metric_col}')
    plot_container = container.expander("Show plot") if hide_plot else container
    col = plot_container.columns([2,3,2])
    col[1].pyplot(fig, use_container_width=False)
    plt.close(fig)


def _eval_seniority(tmp_df, container, metric_col='Mean Distance'):
    result = _compute_seniority(tmp_df, metric_col)
    _render_seniority(result, container, metric_col)
    return result['p']


_DEFINED_DOMAINS = ['Mobile', 'IoT', 'Webentwicklung', 'Infrastruktur', 'Enterprise']
_EMBEDDED_HW_DOMAIN = 'Embedded'
_EMBEDDED_HW_KEYWORDS = ('embedded', 'desktop', 'hardware', 'gpu', 'graphics')
_GAME_DEV_DOMAIN = 'Game Development'
_ALL_DOMAINS = _DEFINED_DOMAINS + [_EMBEDDED_HW_DOMAIN, _GAME_DEV_DOMAIN]


def _classify_domain(val):
    """Primary domain for a participant's (decoded, comma-joined) PQ3 answer: one of the 5
    predefined checkbox categories if selected; else _GAME_DEV_DOMAIN if their free-text
    'Sonstiges' answer was normalized to 'Game Development' by the `mapping` dict (applied to
    `df` before decoding, so PQ3[other] values like 'Gamedev'/'Games'/'GAMEDEV' already read
    'Game Development' here); else _EMBEDDED_HW_DOMAIN if it mentions
    embedded/desktop/hardware/gpu/graphics (the largest remaining cluster of otherwise-
    unclassified free-text answers); else unclassified (pd.NA)."""
    if pd.isna(val):
        return pd.NA
    parts = [p.strip() for p in str(val).split(',')]
    for part in parts:
        if part in _DEFINED_DOMAINS:
            return part
    for part in parts:
        if part == _GAME_DEV_DOMAIN:
            return _GAME_DEV_DOMAIN
    for part in parts:
        if any(kw in part.lower() for kw in _EMBEDDED_HW_KEYWORDS):
            return _EMBEDDED_HW_DOMAIN
    return pd.NA


def _domain_memberships(val):
    """Every domain a participant's (decoded, comma-joined) PQ3 answer belongs to — unlike
    _classify_domain, a participant who named e.g. 'Mobile, IoT' counts toward both groups
    rather than only the first match. Returns a (possibly empty) set."""
    if pd.isna(val):
        return set()
    parts = [p.strip() for p in str(val).split(',')]
    doms = set()
    for part in parts:
        if part in _DEFINED_DOMAINS:
            doms.add(part)
        elif part == _GAME_DEV_DOMAIN:
            doms.add(_GAME_DEV_DOMAIN)
        elif any(kw in part.lower() for kw in _EMBEDDED_HW_KEYWORDS):
            doms.add(_EMBEDDED_HW_DOMAIN)
    return doms


_ENERGY_AWARE_DOMAINS = ('IoT', 'Mobile', _EMBEDDED_HW_DOMAIN)


def _energy_domain_group(val):
    """Splits a participant's (decoded, comma-joined) PQ3 answer into the two independent
    groups compared by the domain Mann-Whitney U test: 'Energy-aware' (IoT, Mobile, or
    Embedded — domains associated with heightened energy awareness) vs. 'Other domains'
    (everything else named, classified or free-text). A participant naming domains from BOTH
    groups is excluded ('Both') to keep the two groups independent; a participant with no PQ3
    answer at all is excluded too (pd.NA)."""
    if pd.isna(val):
        return pd.NA
    parts = [p.strip() for p in str(val).split(',')]
    has_aware = any(
        part in ('IoT', 'Mobile') or any(kw in part.lower() for kw in _EMBEDDED_HW_KEYWORDS)
        for part in parts
    )
    has_other = any(
        not (part in ('IoT', 'Mobile') or any(kw in part.lower() for kw in _EMBEDDED_HW_KEYWORDS))
        for part in parts
    )
    if has_aware and has_other:
        return 'Both'
    if has_aware:
        return 'Energy-aware (IoT/Mobile/Embedded)'
    if has_other:
        return 'Other domains'
    return pd.NA  # empty string / no parts, shouldn't normally happen for a non-NA answer


def _compute_domain(tmp_df, metric_col):
    """Mann-Whitney U comparison of two independent domain groups: participants working in
    domains associated with heightened energy awareness (IoT, Mobile, Embedded) vs. all other
    domains. Participants naming domains from both groups, or no domain at all, are excluded
    to keep the two groups independent — see _energy_domain_group."""
    tmp_df = tmp_df.copy()
    tmp_df = tmp_df.assign(**{metric_col: pd.to_numeric(tmp_df[metric_col], errors='coerce')})

    group_col = tmp_df['PQ3'].apply(_energy_domain_group)
    aware = tmp_df.loc[group_col == 'Energy-aware (IoT/Mobile/Embedded)', metric_col].dropna()
    other = tmp_df.loc[group_col == 'Other domains', metric_col].dropna()

    result = {
        'aware': aware, 'other': other,
        'n_excluded_both': int((group_col == 'Both').sum()),
        'n_excluded_none': int(group_col.isna().sum()),
        'p': None, 'U': None, 'r_rb': None, 'n1': None, 'n2': None,
    }
    if len(aware) == 0 or len(other) == 0:
        return result

    U, p = mannwhitneyu(aware, other, alternative='two-sided')
    n1, n2 = len(aware), len(other)
    r_rb = 2 * U / (n1 * n2) - 1
    result.update({'p': p, 'U': U, 'r_rb': r_rb, 'n1': n1, 'n2': n2})
    return result


def _render_domain(result, container, metric_col, hide_plot=False):
    container.write("#### Analysis 4: Domain (PQ3) — Energy-aware (IoT/Mobile/Embedded) vs. Other domains")
    container.write(_describe_group(result['aware'], 'Energy-aware (IoT/Mobile/Embedded)'))
    container.write(_describe_group(result['other'], 'Other domains'))
    container.write(
        f"Excluded — named domains from both groups: n={result['n_excluded_both']}; "
        f"no PQ3 answer: n={result['n_excluded_none']}"
    )

    if result['p'] is None:
        container.write("Insufficient data for test.")
        return

    container.write(
        f"Mann-Whitney U = {result['U']:.3f}, p = {result['p']:.4f} ({_sig_label(result['p'])}), "
        f"rank-biserial r = {result['r_rb']:.3f}, n₁={result['n1']}, n₂={result['n2']}"
    )

    plot_df = pd.concat([
        result['aware'].rename(metric_col).to_frame().assign(Domain='Energy-aware (IoT/Mobile/Embedded)'),
        result['other'].rename(metric_col).to_frame().assign(Domain='Other domains'),
    ])
    group_labels = ['Energy-aware (IoT/Mobile/Embedded)', 'Other domains']
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.stripplot(data=plot_df, x='Domain', y=metric_col,
                  order=group_labels, jitter=True, alpha=0.6, ax=ax)
    sns.pointplot(data=plot_df, x='Domain', y=metric_col,
                  order=group_labels, estimator=np.median,
                  errorbar=None, markers='D', color='black', ax=ax)
    ax.set_ylabel(metric_col)
    ax.set_title(f'Domain vs. {metric_col}')
    ax.tick_params(axis='x', rotation=10)
    plot_container = container.expander("Show plot") if hide_plot else container
    col = plot_container.columns([2,3,2])
    col[1].pyplot(fig, use_container_width=False)
    plt.close(fig)


def _eval_domain(tmp_df, container, metric_col='Mean Distance'):
    result = _compute_domain(tmp_df, metric_col)
    _render_domain(result, container, metric_col)
    return result['p']


def _prepare_correlation_df(df):
    pq3_category_mapping = {
        'PQ3[SQ001]': 'Mobile',
        'PQ3[SQ002]': 'IoT',
        'PQ3[SQ003]': 'Webentwicklung',
        'PQ3[SQ004]': 'Infrastruktur',
        'PQ3[SQ005]': 'Enterprise'
    }

    pq4_category_mapping = {
        'PQ4[SQ001]': 'Frontend',
        'PQ4[SQ002]': 'Backend',
        'PQ4[SQ003]': 'Full-Stack',
        'PQ4[SQ004]': 'DevOps',
        'PQ4[SQ005]': 'Data/Database',
        'PQ4[SQ006]': 'Design/Architecture'
    }

    scenario1_category_mapping = {
        'Scenario1[SQ001]': 'perfekte positive Korrelation',
        'Scenario1[SQ002]': 'moderate positive Korrelation',
        'Scenario1[SQ003]': 'moderate negative Korrelation',
        'Scenario1[SQ004]': 'perfekte negative Korrelation',
        'Scenario1[SQ005]': 'keine Korrelation'
    }

    scenario2_category_mapping = {
        'Scenario2[SQ001]': 'perfekte positive Korrelation',
        'Scenario2[SQ002]': 'moderate positive Korrelation',
        'Scenario2[SQ003]': 'moderate negative Korrelation',
        'Scenario2[SQ004]': 'perfekte negative Korrelation',
        'Scenario2[SQ005]': 'keine Korrelation'
    }

    scenario3_category_mapping = {
        'Scenario3[SQ001]': 'perfekte Korrelation',
        'Scenario3[SQ002]': 'moderate Korrelation',
        'Scenario3[SQ003]': 'keine Korrelation'
    }

    scenario4_category_mapping = {
        'Scenario4[SQ001]': 'perfekte positive Korrelation',
        'Scenario4[SQ002]': 'moderate positive Korrelation',
        'Scenario4[SQ003]': 'moderate negative Korrelation',
        'Scenario4[SQ004]': 'perfekte negative Korrelation',
        'Scenario4[SQ005]': 'keine Korrelation'
    }

    scale_5pt = [
        'perfekte positive Korrelation',
        'moderate positive Korrelation',
        'keine Korrelation',
        'moderate negative Korrelation',
        'perfekte negative Korrelation',
    ]
    scale_3pt = [
        'perfekte Korrelation',
        'moderate Korrelation',
        'keine Korrelation',
    ]
    correct_answers = {
        'Scenario1': ('perfekte positive Korrelation', scale_5pt),
        'Scenario2': ('moderate positive Korrelation', scale_5pt),
        'Scenario3': ('keine Korrelation', scale_3pt),
        'Scenario4': ('moderate negative Korrelation', scale_5pt),
    }

    tmp_df = df[['id', 'PQ1', 'RQ4']].copy()
    tmp_df.loc[:, 'RQ4'] = tmp_df['RQ4'].replace({
        "Ja, der Energieverbrauch ist relevant in unseren Projekten und wir bewerten den Energieverbrauch unserer Software (z. B. über Energiemessungen, Laufzeitleistung, CPU-Auslastung, Kundenfeedback, Bauchgefühl oder anderes).": "Relevant",
        "Nein, der Energieverbrauch ist nicht relevantfür in unseren Projekten, obwohl wir uns auf andere Eigenschaften (wie Laufzeit oder Cloud-Kosten) konzentrieren, die wiederum den Energieverbrauch beeinflussen können.": "Not relevant"
    })

    tmp_df['PQ2'] = df.apply(
        decode_single_choice_with_other,
        axis=1,
        choice_col='PQ2',
        other_col='PQ2[other]',
        replacements={
            "we don't have a Junior/Senior split, but it would be Senior probably ": "Senior"
        }
    )
    tmp_df['PQ3'] = df.apply(
        decode_multiple_choice_row,
        axis=1,
        category_mapping=pq3_category_mapping,
        other_cols=['PQ3[other]']
    )
    tmp_df['PQ4'] = df.apply(
        decode_multiple_choice_row,
        axis=1,
        category_mapping=pq4_category_mapping,
        other_cols=['PQ4[other]']
    )
    tmp_df['Scenario1'] = df.apply(
        decode_multiple_choice_row,
        axis=1,
        category_mapping=scenario1_category_mapping
    )
    tmp_df['Scenario2'] = df.apply(
        decode_multiple_choice_row,
        axis=1,
        category_mapping=scenario2_category_mapping
    )
    tmp_df['Scenario3'] = df.apply(
        decode_multiple_choice_row,
        axis=1,
        category_mapping=scenario3_category_mapping
    )
    tmp_df['Scenario4'] = df.apply(
        decode_multiple_choice_row,
        axis=1,
        category_mapping=scenario4_category_mapping
    )

    def accuracy_distance(answer_str, correct, scale):
        if pd.isna(answer_str):
            return pd.NA
        correct_idx = scale.index(correct)
        distances = []
        for ans in str(answer_str).split(', '):
            ans = ans.strip()
            if ans in scale:
                distances.append(abs(scale.index(ans) - correct_idx))
        return min(distances) if distances else pd.NA

    for scenario, (correct, scale) in correct_answers.items():
        tmp_df.loc[:, f'Accuracy {scenario}'] = tmp_df[scenario].apply(
            accuracy_distance, correct=correct, scale=scale
        )

    accuracy_cols = [f'Accuracy {s}' for s in correct_answers]
    tmp_df.loc[:, 'Accuracy Total'] = tmp_df[accuracy_cols].sum(axis=1, skipna=False)
    tmp_df.loc[:, 'Mean Distance'] = pd.to_numeric(tmp_df['Accuracy Total'], errors='coerce') / 4

    # Binary accuracy: 0 = correct (distance == 0), 1 = incorrect
    for scenario in correct_answers:
        dist = tmp_df[f'Accuracy {scenario}']
        tmp_df.loc[:, f'Binary {scenario}'] = dist.apply(
            lambda x: 0 if (pd.notna(x) and int(x) == 0) else (1 if pd.notna(x) else pd.NA)
        )

    binary_cols = [f'Binary {s}' for s in correct_answers]
    tmp_df.loc[:, 'Binary Total'] = tmp_df[binary_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1, skipna=False)

    tmp_df = tmp_df.dropna(how='all', subset=tmp_df.columns.difference(['id']))
    # Keep participants who answered at least one of the 4 scenarios — a participant who only
    # answered e.g. Scenario1 must still be included in Scenario1's analysis. Each per-scenario
    # function (_explode_scenario_decisions, the sensitivity strategies, etc.) already does its
    # own dropna/valid-answer filtering on just that scenario's column, so NaNs in the other
    # 3 scenario columns are harmless there. Only rows with ALL 4 scenarios missing are dropped,
    # so the aggregate columns ('Accuracy Total', 'Mean Distance', 'Binary Total' — sums across all
    # 4 scenarios) can legitimately be NaN for a participant who didn't answer every scenario;
    # nothing currently displayed relies on those aggregates being fully populated.
    tmp_df = tmp_df.dropna(how='all', subset=['Scenario1', 'Scenario2', 'Scenario3', 'Scenario4'])

    return tmp_df, scale_5pt, scale_3pt, correct_answers


# Ternary accuracy: 0 = correct, 1 = somehow correct (right direction, wrong strength), 2 = incorrect
_TERNARY_POSITIVE = {'perfekte positive Korrelation', 'moderate positive Korrelation'}
_TERNARY_NEGATIVE = {'perfekte negative Korrelation', 'moderate negative Korrelation'}


def _direction(v):
    if v in _TERNARY_POSITIVE: return 'pos'
    if v in _TERNARY_NEGATIVE: return 'neg'
    return 'none'


def _ternary_score(ans, correct):
    """0 = exact match, 1 = right direction but wrong strength, 2 = otherwise incorrect.
    Applies uniformly to every individual answer, regardless of how many answers the
    participant who gave it selected for this scenario."""
    if ans == correct:
        return 0
    if _direction(ans) == _direction(correct) and _direction(correct) != 'none':
        return 1
    return 2


def _explode_scenario_decisions(tmp_df, scenario, correct, scale):
    """One row per individual selected answer (a decision) rather than per participant.

    A participant who selected multiple options for a scenario contributes one row per
    selection, each scored independently — no min()-collapsing to a single best answer.
    """
    covariate_cols = ['id', 'PQ1', 'RQ4', 'PQ2', 'PQ3', 'PQ4']
    sub = tmp_df[covariate_cols + [scenario]].dropna(subset=[scenario]).copy()
    sub.loc[:, scenario] = sub[scenario].astype(str).str.split(', ')
    sub = sub.explode(scenario)
    sub.loc[:, scenario] = sub[scenario].str.strip()
    sub = sub[sub[scenario].isin(scale)].copy()
    sub.loc[:, f'Ternary {scenario}'] = sub[scenario].apply(_ternary_score, correct=correct)
    return sub.reset_index(drop=True)


def _valid_answers(answer_str, scale):
    if pd.isna(answer_str):
        return []
    return [a.strip() for a in str(answer_str).split(', ') if a.strip() in scale]


def _answer_magnitude(answer):
    """Higher = more extreme. 'perfekte ...' > 'moderate ...' > 'keine Korrelation'."""
    if answer.startswith('perfekte'):
        return 2
    if answer.startswith('moderate'):
        return 1
    return 0


def _answer_family(answer):
    """Groups an answer for contradiction-checking.

    5-point scale (Scenario1/2/4): 'pos' / 'neg' / 'none' (via _direction).
    3-point scale (Scenario3, no pos/neg tags): 'has_corr' for 'perfekte'/'moderate Korrelation',
    'none' for 'keine Korrelation'. Two selections contradict each other if they fall into more
    than one family (e.g. positive + negative, or no-correlation + positive/negative/has-correlation).
    """
    d = _direction(answer)
    if d != 'none':
        return d
    if answer == 'keine Korrelation':
        return 'none'
    return 'has_corr'


def _strategy_single_answer_only(tmp_df, scenario, correct, scale):
    """Strategy 1: keep only participants who gave exactly one valid answer for this scenario."""
    covariate_cols = ['id', 'PQ1', 'RQ4', 'PQ2', 'PQ3', 'PQ4']
    sub = tmp_df[covariate_cols + [scenario]].copy()
    answers = sub[scenario].apply(lambda s: _valid_answers(s, scale))
    mask = answers.apply(len) == 1

    sub = sub[mask].copy()
    sub.loc[:, scenario] = answers[mask].apply(lambda lst: lst[0])
    sub.loc[:, f'Ternary {scenario}'] = sub[scenario].apply(_ternary_score, correct=correct)
    return sub.reset_index(drop=True)


def _strategy_resolve_noncontradicting(tmp_df, scenario, correct, scale):
    """Strategy 2: keep single-answer participants as-is. For participants who selected multiple
    answers, resolve to the most extreme one if all selections share one family (same direction, or
    all 'has correlation' on the 3-point scale); exclude the participant entirely if their
    selections contradict each other (mixing positive/negative/no-correlation families)."""
    covariate_cols = ['id', 'PQ1', 'RQ4', 'PQ2', 'PQ3', 'PQ4']
    sub = tmp_df[covariate_cols + [scenario]].copy()
    valid = sub[scenario].apply(lambda s: _valid_answers(s, scale))

    def resolve(answers):
        if len(answers) == 0:
            return pd.NA
        if len(answers) == 1:
            return answers[0]
        if len({_answer_family(a) for a in answers}) > 1:
            return pd.NA
        return max(answers, key=_answer_magnitude)

    sub.loc[:, scenario] = valid.apply(resolve)
    sub = sub.dropna(subset=[scenario]).copy()
    sub.loc[:, f'Ternary {scenario}'] = sub[scenario].apply(_ternary_score, correct=correct)
    return sub.reset_index(drop=True)


def _strategy2_scenario_likert_data(df):
    """Per-scenario Likert counts (1 = perfect negative ... 5 = perfect positive) for the Likert
    plots, built from the Strategy 2 sample (one resolved answer per participant, contradicting
    participants excluded) so they match the other reported results.

    Scenario3 uses the 3-point scale; its answers are placed on 3 (none) / 4 (moderate) / 5 (perfect).

    Returns (overall, by_relevance): `overall` maps scenario label -> counts over all remaining
    participants; `by_relevance` maps label -> counts for participants who consider energy
    consumption not relevant, and "<label> (R)" -> counts for those who do.
    """
    tmp_df, scale_5pt, _, correct_answers = _prepare_correlation_df(df)
    likert_value = {a: i + 1 for i, a in enumerate(reversed(scale_5pt))}
    likert_value.update({'keine Korrelation': 3, 'moderate Korrelation': 4, 'perfekte Korrelation': 5})
    labels = {
        'Scenario3': 'No correlation',
        'Scenario4': 'Negative correlation',
        'Scenario2': 'Moderate positive correlation',
        'Scenario1': 'Strong positive correlation',
    }

    def likert_counts(answers):
        values = answers.map(likert_value)
        return {v: int((values == v).sum()) for v in range(1, 6)}

    overall, by_relevance = {}, {}
    for scenario, label in labels.items():
        correct, scale = correct_answers[scenario]
        resolved = _strategy_resolve_noncontradicting(tmp_df, scenario, correct, scale)
        relevant = resolved['RQ4'] == 'Relevant'
        overall[label] = likert_counts(resolved[scenario])
        by_relevance[label] = likert_counts(resolved.loc[~relevant, scenario])
        by_relevance[f'{label} (R)'] = likert_counts(resolved.loc[relevant, scenario])
    return overall, by_relevance


def _strategy_random_multi_resolution(tmp_df, scenario, correct, scale, target_repetitions=30, seed=42):
    """Strategy 3: keep single-answer participants unchanged. For participants who selected
    multiple answers, randomly pick one of their given answers per repetition. Repetitions are
    deduplicated (no two repetitions resolve the multi-answer participants identically) and the
    repetition count is capped at however many distinct combinations actually exist. Deterministic
    given `seed`.

    Returns (repetitions, n_used, n_requested) where `repetitions` is a list of resolved
    DataFrames (one row per participant, with a `Ternary {scenario}` column each).
    """
    covariate_cols = ['id', 'PQ1', 'RQ4', 'PQ2', 'PQ3', 'PQ4']
    sub = tmp_df[covariate_cols + [scenario]].copy()
    valid = sub[scenario].apply(lambda s: _valid_answers(s, scale))

    single_mask = valid.apply(len) == 1
    multi_mask = valid.apply(len) > 1

    base_df = sub[single_mask].copy()
    base_df.loc[:, scenario] = valid[single_mask].apply(lambda lst: lst[0])

    multi_df = sub[multi_mask].copy()
    multi_options = valid[multi_mask].tolist()

    def _finalize(resolved_answers_df):
        resolved_answers_df = resolved_answers_df.copy()
        resolved_answers_df.loc[:, f'Ternary {scenario}'] = resolved_answers_df[scenario].apply(
            _ternary_score, correct=correct
        )
        return resolved_answers_df.reset_index(drop=True)

    if len(multi_options) == 0:
        return [_finalize(base_df)], 1, target_repetitions

    total_combinations = 1
    for opts in multi_options:
        total_combinations *= len(opts)

    n_repetitions = min(target_repetitions, total_combinations)
    rng = random.Random(seed)

    if total_combinations <= 200_000:
        all_combos = list(itertools.product(*multi_options))
        rng.shuffle(all_combos)
        chosen_combos = all_combos[:n_repetitions]
    else:
        seen = set()
        chosen_combos = []
        while len(chosen_combos) < n_repetitions:
            combo = tuple(rng.choice(opts) for opts in multi_options)
            if combo not in seen:
                seen.add(combo)
                chosen_combos.append(combo)

    repetitions = []
    for combo in chosen_combos:
        rep_multi = multi_df.copy()
        rep_multi.loc[:, scenario] = list(combo)
        repetitions.append(_finalize(pd.concat([base_df, rep_multi], ignore_index=True)))

    return repetitions, n_repetitions, target_repetitions


def _summarize_p_values(p_values):
    valid_ps = [p for p in p_values if p is not None]
    if not valid_ps:
        return None
    arr = np.array(valid_ps)
    return {
        'n_valid': len(valid_ps),
        'n_total': len(p_values),
        'min': float(arr.min()),
        'median': float(np.median(arr)),
        'max': float(arr.max()),
        'pct_sig': float((arr < 0.05).mean() * 100),
    }


_SENSITIVITY_ANALYSES = [
    ('Expert vs. Non-Expert (RQ4)', 'p'),
    ('Years of Experience (PQ1)', 'p'),
    ('Seniority (PQ2)', 'p'),
    ('Domain (PQ3)', 'p'),
    ('Ordinal Logistic Regression', 'min p'),
]


def _sensitivity_analysis_specs(metric_col):
    return [
        ('Expert vs. Non-Expert (RQ4)', lambda df: _compute_expert_vs_nonexpert(df, metric_col)['p'], 'p'),
        ('Years of Experience (PQ1)', lambda df: _compute_years_experience(df, metric_col)['p'], 'p'),
        ('Seniority (PQ2)', lambda df: _compute_seniority(df, metric_col)['p'], 'p'),
        ('Domain (PQ3)', lambda df: _compute_domain(df, metric_col)['p'], 'p'),
        ('Ordinal Logistic Regression', lambda df: _compute_ordinal_logistic_regression(df, metric_col)['min_p'], 'min p'),
    ]


def _render_sensitivity_strategy3_summary(tmp_df, scenario, correct, scale, container,
                                           target_repetitions=30, seed=42):
    """Renders the aggregate Strategy-3 summary and returns the full per-repetition p-value
    lists as {analysis name: [p, ...]} for use in cross-strategy comparisons."""
    metric_col = f'Ternary {scenario}'
    repetitions, n_used, n_requested = _strategy_random_multi_resolution(
        tmp_df, scenario, correct, scale, target_repetitions=target_repetitions, seed=seed
    )

    cap_note = " (capped — fewer unique combinations exist)" if n_used < n_requested else ""
    container.write(f"Repetitions: {n_used} / {n_requested} requested{cap_note}")
    container.write(f"n = {len(repetitions[0])} participants per repetition")

    dist_counts = {0: [], 1: [], 2: []}
    for rep_df in repetitions:
        counts = rep_df[metric_col].value_counts()
        for v in (0, 1, 2):
            dist_counts[v].append(int(counts.get(v, 0)))
    dist_summary = ", ".join(
        f"{v}: {min(c)}–{max(c)} (mean {np.mean(c):.1f})" for v, c in dist_counts.items()
    )
    container.write("Ternary Score distribution across repetitions — " + dist_summary)

    p_values_by_analysis = {}
    for name, fn, p_label in _sensitivity_analysis_specs(metric_col):
        p_values = [fn(rep_df) for rep_df in repetitions]
        p_values_by_analysis[name] = p_values

        summary = _summarize_p_values(p_values)
        exp = container.expander(name)
        if summary is None:
            exp.write("Insufficient data in all repetitions.")
            continue
        exp.write(
            f"Valid in {summary['n_valid']}/{summary['n_total']} repetitions. "
            f"{p_label}: min = {summary['min']:.4f}, median = {summary['median']:.4f}, "
            f"max = {summary['max']:.4f}. Significant ({p_label} < 0.05) in "
            f"{summary['pct_sig']:.0f}% of repetitions."
        )

    return p_values_by_analysis


def _expander_title(name, p, p_label='p'):
    if p is None:
        return f"{name} — insufficient data"
    return f"{name} — {p_label} = {p:.4f} ({_sig_label(p)})"


def _significance_expander(container, key, label, p):
    """A plain st.expander. Previously highlighted lightgreen when p < 0.05, but none of these
    results survive Benjamini-Hochberg correction across the table (see
    _render_strategy2_pvalue_summary), so highlighting raw-p significance here would be
    misleading and has been removed."""
    wrapper = container.container(key=key)
    return wrapper.expander(label)


def _run_ordinal_analyses(df_subset, container, metric_col):
    _eval_expert_vs_nonexpert(df_subset, container, metric_col)
    _eval_years_experience(df_subset, container, metric_col)
    _eval_seniority(df_subset, container, metric_col)
    _eval_domain(df_subset, container, metric_col)


_SEPARATION_T_THRESHOLD = 50  # |t| beyond this indicates (quasi-)complete separation, not a real effect


# rpy2's Python<->R conversion rules live in a contextvars.ContextVar, which is thread-local and
# is only populated in whichever thread first imports/uses rpy2 (also the thread that embeds R —
# R itself can only safely be driven from one OS thread for the life of the process). Streamlit
# runs each session/rerun's script on its own thread, so a naive rpy2 call from a later
# session/rerun thread fails with "Conversion rules ... appear to be missing". Funneling every
# rpy2/R call through this single dedicated worker thread avoids that entirely — but only if the
# executor itself survives reruns. Streamlit re-executes the whole script top-to-bottom on every
# rerun, so a plain module-level assignment would spin up a *new* ThreadPoolExecutor (and thus a
# new, uninitialized worker thread) each time. st.cache_resource makes it a process-wide singleton
# instead, so the same worker thread — and its already-populated conversion rules — is reused.
@st.cache_resource
def _get_r_executor():
    return concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="rpy2-worker")

def _fit_ordinal_regression_in_r(valid, predictor_cols):
    """Must only ever be called via _get_r_executor().submit(...) — see the note above."""
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, default_converter
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    # importr() probes R's default library paths and warns if some are empty — harmless here,
    # since MASS loads from wherever it actually is. Suppress just for this call so we don't lose
    # warnings raised later (e.g. from polr/confint on genuinely ill-conditioned fits).
    ro.r('options(warn = -1)')
    importr('MASS')
    ro.r('options(warn = 0)')

    with localconverter(default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(valid.astype(float))
    ro.globalenv['r_df'] = r_df
    ro.r('r_df$outcome <- factor(r_df$outcome, levels=c(0,1,2), ordered=TRUE)')

    formula_str = 'outcome ~ ' + ' + '.join(predictor_cols)
    ro.r(f'model    <- polr({formula_str}, data=r_df, Hess=TRUE)')
    ro.r('ctable    <- coef(summary(model))')
    ro.r('p_vals    <- pnorm(abs(ctable[, "t value"]), lower.tail=FALSE) * 2')
    ro.r('''
ci_method <- "Profile likelihood"
ci <- tryCatch(
    # confint()'s internal approx() warns when the profile-likelihood grid has duplicate x-values
    # (common on small samples) — cosmetic, suppressed here; a genuine failure still falls through
    # to the Wald fallback below via the error handler.
    suppressWarnings(suppressMessages(confint(model))),
    error = function(e) {
        ci_method <<- "Wald (profile CI failed)"
        suppressWarnings(confint.default(model))
    }
)
    ''')
    ro.r('OR        <- exp(coef(model))')
    ro.r('OR_ci    <- exp(ci)')
    ro.r('null_mod <- polr(outcome ~ 1, data=r_df, Hess=TRUE)')

    n_pred    = len(predictor_cols)
    ctable_np = np.array(ro.r('ctable'))
    p_np      = np.array(ro.r('p_vals'))
    ci_np     = np.array(ro.r('ci'))
    OR_np     = np.array(ro.r('OR'))
    OR_ci_np  = np.array(ro.r('OR_ci'))
    ll_full   = float(ro.r('as.numeric(logLik(model))')[0])
    ll_null   = float(ro.r('as.numeric(logLik(null_mod))')[0])
    aic_val   = float(ro.r('AIC(model)')[0])
    ci_method = str(ro.r('ci_method')[0])

    return {
        'n_pred': n_pred, 'ctable_np': ctable_np, 'p_np': p_np, 'ci_np': ci_np,
        'OR_np': OR_np, 'OR_ci_np': OR_ci_np, 'll_full': ll_full, 'll_null': ll_null,
        'aic_val': aic_val, 'ci_method': ci_method,
    }


def _compute_ordinal_logistic_regression(tmp_df, metric_col):
    reg_df = tmp_df.copy()
    outcome = pd.to_numeric(reg_df[metric_col], errors='coerce')
    expertise = reg_df['RQ4'].map({'Relevant': 1, 'Not relevant': 0})
    years = pd.to_numeric(reg_df['PQ1'], errors='coerce')

    _seniority_ordinal = {'Junior': 0, 'Mid': 1, 'Senior': 2, 'Lead': 3}

    def _map_seniority_ordinal(val):
        lvl = _classify_seniority(val)
        return _seniority_ordinal.get(lvl, np.nan)

    seniority = reg_df['PQ2'].apply(_map_seniority_ordinal)

    domain = reg_df['PQ3'].apply(_classify_domain)
    domain_dummies = pd.get_dummies(domain, prefix='Dom', drop_first=True).astype(float)

    X = pd.DataFrame({'Expertise': expertise, 'Years': years, 'Seniority': seniority})
    X = pd.concat([X, domain_dummies], axis=1)
    X.columns = [c.replace(' ', '_').replace('/', '_') for c in X.columns]

    valid = pd.concat([outcome.rename('outcome'), X], axis=1).dropna()
    predictor_cols = list(X.columns)

    result = {
        'valid': valid, 'predictor_cols': predictor_cols, 'error': None,
        'summary_df': None, 'mcfadden': None, 'll_full': None, 'aic_val': None,
        'ci_method': None, 'min_p': None,
    }

    if len(valid) < 10:
        result['error'] = "Insufficient data for ordinal logistic regression."
        return result
    if valid['outcome'].nunique() < 2:
        result['error'] = "Outcome has fewer than 2 distinct categories — cannot fit model."
        return result

    try:
        r_out = _get_r_executor().submit(_fit_ordinal_regression_in_r, valid, predictor_cols).result()

        n_pred    = r_out['n_pred']
        ctable_np = r_out['ctable_np']
        p_np      = r_out['p_np']
        OR_np     = r_out['OR_np']
        OR_ci_np  = r_out['OR_ci_np']
        ll_full   = r_out['ll_full']
        ll_null   = r_out['ll_null']
        aic_val   = r_out['aic_val']
        ci_method = r_out['ci_method']

        summary_df = pd.DataFrame({
            'Coef (log-OR)': ctable_np[:n_pred, 0],
            'Std Err':        ctable_np[:n_pred, 1],
            't':              ctable_np[:n_pred, 2],
            'p-value':        p_np[:n_pred],
            'OR':             OR_np,
            'OR 2.5%':        OR_ci_np[:, 0],
            'OR 97.5%':       OR_ci_np[:, 1],
        }, index=predictor_cols)
        summary_df.loc[:, 'Sig'] = summary_df['p-value'].apply(_sig_label)

        # (Quasi-)complete separation: a predictor whose |t| is implausibly large (standard error
        # collapsed toward 0) usually means a small subgroup shares one identical outcome, not a
        # real effect — polr's MLE diverges rather than converging to a finite estimate. Flag and
        # exclude these from min_p so a degenerate fit doesn't masquerade as a significant result.
        summary_df.loc[:, 'Separated'] = summary_df['t'].abs() > _SEPARATION_T_THRESHOLD
        reliable_p = summary_df.loc[~summary_df['Separated'], 'p-value']

        mcfadden = 1 - ll_full / ll_null
        result.update({
            'summary_df': summary_df, 'mcfadden': mcfadden, 'll_full': ll_full,
            'aic_val': aic_val, 'ci_method': ci_method,
            'min_p': float(reliable_p.min()) if len(reliable_p) else None,
        })
    except Exception as e:
        result['error'] = f"rpy2/MASS failed: {e}"

    return result


def _render_ordinal_logistic_regression(result, container):
    container.write("#### Ordinal Logistic Regression (MASS::polr via rpy2)")
    container.write(
        "Predictors: Expertise (RQ4), Years of Experience (PQ1), Seniority (PQ2), Domain (PQ3). "
        "Outcome: 0 = correct, 1 = partially correct (right direction), 2 = incorrect. "
        "P-values: normal approximation on t-values. CIs: profile likelihood (confint)."
    )

    if result['summary_df'] is None:
        container.write(result['error'])
        return

    valid = result['valid']
    container.write(f"n = {len(valid)}, outcome distribution: {valid['outcome'].value_counts().sort_index().to_dict()}")
    container.dataframe(result['summary_df'].round(4))

    n_separated = int(result['summary_df']['Separated'].sum())
    if n_separated:
        separated_names = ', '.join(result['summary_df'].index[result['summary_df']['Separated']])
        container.write(
            f"_Note: {n_separated} predictor(s) — {separated_names} — show signs of "
            f"(quasi-)complete separation (|t| > {_SEPARATION_T_THRESHOLD}, standard error collapses "
            f"toward 0), typically because a small subgroup's outcomes are all identical. Their "
            f"coefficients are not reliable and are excluded from the min-p summary above, but still "
            f"shown in the table for transparency._"
        )

    container.write(
        f"Pseudo R² (McFadden): {result['mcfadden']:.4f} | "
        f"Log-Likelihood: {result['ll_full']:.3f} | AIC: {result['aic_val']:.3f} | "
        f"CIs: {result['ci_method']}"
    )


def _eval_ordinal_logistic_regression(tmp_df, container, metric_col):
    result = _compute_ordinal_logistic_regression(tmp_df, metric_col)
    _render_ordinal_logistic_regression(result, container)
    return result['min_p']


def correlation_analysis(df, container):
    container.write(
        "##### Extended Correlation Analysis: Wie korrelieren die Antworten zu den Fragen PQ1, PQ2, PQ3, PQ4 mit den Antworten zu den Szenarien Scenario1, Scenario2, Scenario3 und Scenario4?"
    )

    tmp_df, scale_5pt, scale_3pt, correct_answers = _prepare_correlation_df(df)

    container.write(tmp_df)
    container.write("Number Participants: " + str(len(tmp_df)))

    st.write(
        "**Ternary Accuracy** — 0 = correct, 1 = somehow correct (right direction, wrong strength), "
        "2 = incorrect. Applies uniformly to every individual answer. Granularity: **decisions**, "
        "not participants — a participant who selected multiple options for a scenario contributes "
        "one observation per selection (no min()-collapsing to their single best answer). "
        "Tests: Mann-Whitney U (2 groups), Kruskal-Wallis (k groups), Spearman ρ (years)."
    )

    scenario_labels = {
        'Scenario1': 'Scenario 1 (correct: perfekte positive)',
        'Scenario2': 'Scenario 2 (correct: moderate positive)',
        'Scenario3': 'Scenario 3 (correct: keine Korrelation)',
        'Scenario4': 'Scenario 4 (correct: moderate negative)',
    }
    sc_tab_labels = list(scenario_labels.values())

    decisions_by_scenario = {
        scenario: _explode_scenario_decisions(tmp_df, scenario, correct, scale)
        for scenario, (correct, scale) in correct_answers.items()
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharey=True)
    for ax, (scenario, label) in zip(axes, scenario_labels.items()):
        data = decisions_by_scenario[scenario][f'Ternary {scenario}']
        sns.histplot(data, bins=[-0.5, 0.5, 1.5, 2.5], discrete=True, ax=ax)
        ax.set_xticks([0, 1, 2])
        ax.set_xlabel('Ternary Score')
        ax.set_title(label)
    axes[0].set_ylabel('Decisions')
    fig.suptitle('Distribution of Ternary Score per Scenario (0 = correct, 1 = right direction, 2 = incorrect)')
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    sc_tabs = container.tabs(sc_tab_labels)
    for tab, scenario in zip(sc_tabs, scenario_labels):
        scenario_df = decisions_by_scenario[scenario]
        tab.write(f"**n = {len(scenario_df)} decisions**")
        _run_ordinal_analyses(scenario_df, tab, f'Ternary {scenario}')
        _eval_ordinal_logistic_regression(scenario_df, tab, f'Ternary {scenario}')


def _render_sensitivity_strategy_results(strat_df, scenario, container, key_prefix):
    """Renders the 5 sub-analyses for one strategy's resolved data and returns their p-values
    as {analysis name: p} for use in cross-strategy comparisons. `key_prefix` must be unique per
    (scenario, strategy) call site — it seeds the CSS keys used to highlight significant results."""
    metric_col = f'Ternary {scenario}'
    container.write(f"n = {len(strat_df)} participants")
    if len(strat_df) == 0:
        container.write("_No data remaining under this strategy._")
        return {name: None for name, _ in _SENSITIVITY_ANALYSES}

    dist = strat_df[metric_col].value_counts().sort_index()
    container.write("Ternary Score distribution — " + ", ".join(f"{int(k)}: {v}" for k, v in dist.items()))

    stats1 = _compute_expert_vs_nonexpert(strat_df, metric_col)
    exp1 = _significance_expander(container, f"{key_prefix}_expert", _expander_title("Expert vs. Non-Expert (RQ4)", stats1['p']), stats1['p'])
    _render_expert_vs_nonexpert(stats1, exp1, metric_col, hide_plot=True)

    stats2 = _compute_years_experience(strat_df, metric_col)
    exp2 = _significance_expander(container, f"{key_prefix}_years", _expander_title("Years of Experience (PQ1)", stats2['p']), stats2['p'])
    _render_years_experience(stats2, exp2, metric_col, hide_plot=True)

    stats3 = _compute_seniority(strat_df, metric_col)
    exp3 = _significance_expander(container, f"{key_prefix}_seniority", _expander_title("Seniority (PQ2)", stats3['p']), stats3['p'])
    _render_seniority(stats3, exp3, metric_col, hide_plot=True)

    stats4 = _compute_domain(strat_df, metric_col)
    exp4 = _significance_expander(container, f"{key_prefix}_domain", _expander_title("Domain (PQ3)", stats4['p']), stats4['p'])
    _render_domain(stats4, exp4, metric_col, hide_plot=True)

    stats5 = _compute_ordinal_logistic_regression(strat_df, metric_col)
    exp5 = _significance_expander(container, f"{key_prefix}_regression", _expander_title("Ordinal Logistic Regression", stats5['min_p'], p_label='min p'), stats5['min_p'])
    _render_ordinal_logistic_regression(stats5, exp5)

    return {
        'Expert vs. Non-Expert (RQ4)': stats1['p'],
        'Years of Experience (PQ1)': stats2['p'],
        'Seniority (PQ2)': stats3['p'],
        'Domain (PQ3)': stats4['p'],
        'Ordinal Logistic Regression': stats5['min_p'],
    }


def _render_strategy_pvalue_comparison(p1_by_analysis, p2_by_analysis, p3_by_analysis, container):
    """One subplot per analysis function: a histogram of Strategy 3's p-values across all
    repetitions, with Strategy 1's and Strategy 2's single p-value marked as vertical lines."""
    container.write("**P-value comparison across strategies**")

    names = [name for name, _ in _SENSITIVITY_ANALYSES]
    fig, axes = plt.subplots(1, len(names), figsize=(7 * len(names), 6), dpi=150, sharex=True)

    for ax, name in zip(axes, names):
        p3_values = [p for p in p3_by_analysis.get(name, []) if p is not None]
        if p3_values:
            sns.histplot(p3_values, bins=15, ax=ax, color='#9fb8d9', edgecolor='white')
        else:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes, fontsize=14)

        ax.axvline(0.05, color='grey', linestyle=':', linewidth=1.5)

        p1 = p1_by_analysis.get(name)
        p2 = p2_by_analysis.get(name)
        if p1 is not None:
            ax.axvline(p1, color='#d95f02', linestyle='--', linewidth=3, label=f'Strategy 1 (p={p1:.3f})')
        if p2 is not None:
            ax.axvline(p2, color='#1b9e77', linestyle='--', linewidth=3, label=f'Strategy 2 (p={p2:.3f})')

        ax.set_xlim(0, 1)
        ax.set_xlabel('p-value', fontsize=13)
        ax.set_ylabel(ax.get_ylabel(), fontsize=13)
        ax.tick_params(axis='both', labelsize=11)
        ax.set_title(name.replace(' (', '\n('), fontsize=15)
        if p1 is not None or p2 is not None:
            ax.legend(fontsize=11)

    axes[0].set_ylabel('Repetitions (Strategy 3)', fontsize=13)
    fig.suptitle(
        'P-value distribution across strategies (bars = Strategy 3 repetitions; '
        'dashed lines = Strategy 1/2; dotted line = p=0.05)',
        fontsize=15
    )
    fig.tight_layout()
    container.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _benjamini_hochberg(pvals):
    """BH-adjusted p-values (step-up FDR correction). None entries pass through as None;
    the correction is computed over the remaining values as a single family of tests."""
    pairs = [(i, p) for i, p in enumerate(pvals) if p is not None]
    if not pairs:
        return [None] * len(pvals)

    idx, ps = zip(*pairs)
    ps = np.array(ps, dtype=float)
    order = np.argsort(ps)
    ranked = ps[order]
    m = len(ranked)

    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]  # enforce monotonicity
    adj = np.clip(adj, 0, 1)

    result = [None] * len(pvals)
    for rank_pos in range(m):
        result[idx[order[rank_pos]]] = adj[rank_pos]
    return result


def _render_strategy2_pvalue_summary(tmp_df, correct_answers, container):
    """Overview table of Strategy 2 p-values and participant counts across the four scenarios.
    p-values are Benjamini-Hochberg corrected as one family across the whole table (all
    characteristics x all scenarios); each cell shows the raw p-value with the corrected value
    in brackets."""
    scenario_order = ['Scenario1', 'Scenario2', 'Scenario4', 'Scenario3']
    display_labels = {
        'Scenario1': 'Strong pos.',
        'Scenario2': 'Moderate pos.',
        'Scenario4': 'Negative',
        'Scenario3': 'No corr.',
    }
    columns = [display_labels[s] for s in scenario_order]

    p_row_names = [
        'Expertise (MW U)',
        'Years of exp. (Sp. ρ)',
        'Seniority (KW H)',
        'Domain (MW U)',
    ]
    n_by_col = {}
    raw_p = {name: {} for name in p_row_names}

    for scenario in scenario_order:
        correct, scale = correct_answers[scenario]
        strat2_df = _strategy_resolve_noncontradicting(tmp_df, scenario, correct, scale)
        metric_col = f'Ternary {scenario}'
        col = display_labels[scenario]

        n_by_col[col] = len(strat2_df)
        if len(strat2_df) == 0:
            for name in p_row_names:
                raw_p[name][col] = None
            continue

        stats_expertise = _compute_expert_vs_nonexpert(strat2_df, metric_col)
        stats_years = _compute_years_experience(strat2_df, metric_col)
        stats_seniority = _compute_seniority(strat2_df, metric_col)
        stats_domain = _compute_domain(strat2_df, metric_col)

        raw_p['Expertise (MW U)'][col] = stats_expertise['p']
        raw_p['Years of exp. (Sp. ρ)'][col] = stats_years['p']
        raw_p['Seniority (KW H)'][col] = stats_seniority['p']
        raw_p['Domain (MW U)'][col] = stats_domain['p']

    # BH correction over the whole table as one family of tests
    flat_cells = [(name, col) for name in p_row_names for col in columns]
    flat_p = [raw_p[name][col] for name, col in flat_cells]
    flat_adj = _benjamini_hochberg(flat_p)
    adj_p = {name: {} for name in p_row_names}
    for (name, col), a in zip(flat_cells, flat_adj):
        adj_p[name][col] = a

    def fmt_cell(p, a):
        if p is None:
            return "—"
        return f"{p:.3f} [{a:.3f}]"

    rows = {'Number of participants': {col: f"n={n_by_col[col]}" for col in columns}}
    for name in p_row_names:
        rows[name] = {col: fmt_cell(raw_p[name][col], adj_p[name][col]) for col in columns}

    summary_df = pd.DataFrame(rows).T[columns]
    summary_df.index.name = 'Characteristic'

    container.write("**Strategy 2 overview: p-values and participant counts across scenarios**")
    container.write(
        "Figures below use Strategy 2 (resolve non-contradicting multi-answers to their most "
        "extreme selection; exclude participants whose selections contradict each other). "
        "Participant counts therefore vary by scenario, since the number excluded for "
        "contradicting answers differs per scenario. p-values are shown as raw [Benjamini-Hochberg "
        "corrected], corrected jointly across all cells of this table."
    )
    container.table(summary_df)

    latex_lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Characteristic} &",
        r"  \textbf{Strong pos.} &",
        r"  \textbf{Moderate pos.} &",
        r"  \textbf{Negative} &",
        r"  \textbf{No corr.} \\",
        r"\midrule",
        f"Number of participants          & n={n_by_col['Strong pos.']} & n={n_by_col['Moderate pos.']} "
        f"& n={n_by_col['Negative']} & n={n_by_col['No corr.']} \\\\",
        r"\midrule",
    ]

    def latex_cell(p, a):
        if p is None:
            return "---"
        return f"${a:.3f}$"

    latex_row_labels = {
        'Expertise (MW U)': 'Expertise (MW U)',
        'Years of exp. (Sp. ρ)': r"Years of exp.\ (Sp.\ $\rho$)",
        'Seniority (KW H)': 'Seniority (KW H)',
        'Domain (MW U)': 'Domain (MW U)',
    }
    for name in p_row_names:
        label = latex_row_labels[name]
        cells = " & ".join(latex_cell(raw_p[name][col], adj_p[name][col]) for col in columns)
        latex_lines.append(f"{label.ljust(32)} & {cells} \\\\")

    latex_lines += [r"\bottomrule", r"\end{tabular}%"]
    latex_table = "\n".join(latex_lines)

    latex_expander = container.expander("LaTeX")
    latex_expander.code(latex_table, language="latex")


def _render_strategy2_subgroup_sizes(tmp_df, correct_answers, container):
    """Subgroup sizes (n) feeding each test in the p-value table above, per scenario.
    Values come from the same 'groups'/'expert'/'non_expert' data the Strategy-2 detail
    expanders below use (e.g. Analysis 3: Seniority) — this just tabulates their sizes side
    by side across scenarios instead of spelling them out per scenario tab."""
    scenario_order = ['Scenario1', 'Scenario2', 'Scenario4', 'Scenario3']
    display_labels = {
        'Scenario1': 'Strong pos.',
        'Scenario2': 'Moderate pos.',
        'Scenario4': 'Negative',
        'Scenario3': 'No corr.',
    }
    columns = [display_labels[s] for s in scenario_order]

    seniority_order = ['Junior', 'Senior', 'Lead']
    domain_row_names = [
        'Domain — Energy-aware (IoT/Mobile/Embedded)',
        'Domain — Other domains',
        'Domain — Excluded (named both groups)',
        'Domain — Excluded (no PQ3 answer)',
    ]

    row_order = (
        ['Expertise — Relevant (Expert)', 'Expertise — Not relevant (Non-Expert)']
        + [f'Seniority — {lvl}' for lvl in seniority_order]
        + ['Seniority — Non-Lead (Junior+Senior, combined)']
        + ['Seniority — Total (classified)']
        + domain_row_names
    )
    rows = {name: {col: 0 for col in columns} for name in row_order}

    for scenario in scenario_order:
        correct, scale = correct_answers[scenario]
        strat2_df = _strategy_resolve_noncontradicting(tmp_df, scenario, correct, scale)
        metric_col = f'Ternary {scenario}'
        col = display_labels[scenario]
        if len(strat2_df) == 0:
            continue

        stats_expertise = _compute_expert_vs_nonexpert(strat2_df, metric_col)
        rows['Expertise — Relevant (Expert)'][col] = len(stats_expertise['expert'])
        rows['Expertise — Not relevant (Non-Expert)'][col] = len(stats_expertise['non_expert'])

        stats_seniority = _compute_seniority(strat2_df, metric_col)
        for lvl in seniority_order:
            rows[f'Seniority — {lvl}'][col] = len(stats_seniority['groups'].get(lvl, []))
        non_lead_n = sum(len(stats_seniority['groups'].get(lvl, [])) for lvl in ['Junior', 'Senior'])
        rows['Seniority — Non-Lead (Junior+Senior, combined)'][col] = non_lead_n
        rows['Seniority — Total (classified)'][col] = sum(len(g) for g in stats_seniority['groups'].values())

        stats_domain = _compute_domain(strat2_df, metric_col)
        rows['Domain — Energy-aware (IoT/Mobile/Embedded)'][col] = len(stats_domain['aware'])
        rows['Domain — Other domains'][col] = len(stats_domain['other'])
        rows['Domain — Excluded (named both groups)'][col] = stats_domain['n_excluded_both']
        rows['Domain — Excluded (no PQ3 answer)'][col] = stats_domain['n_excluded_none']

    subgroup_df = pd.DataFrame(rows).T[columns]
    subgroup_df.index.name = 'Characteristic'

    container.write("**Subgroup sizes across scenarios (Strategy 2)**")
    container.write(
        "Group sizes feeding the tests above, per scenario. Expertise always covers all "
        "participants (RQ4 is a mandatory filter question). 'Mid' seniority has no row of its "
        "own: raw 'Mid'/'Mid Level' answers are mapped to 'Senior' by the `mapping` dict, so no "
        "participant is ever classified as Mid (n = 0) and 'Non-Lead' is exactly Junior + Senior. "
        "Years of experience isn't shown here (continuous predictor, no subgroups).\n\n"
        "**Seniority — 6 participants always unclassified** (same 6 across all scenarios, since "
        "their PQ2 answer doesn't vary by scenario): P21 and P55 left PQ2 empty; P104 answered "
        "'Sole'; P37 ('Small team without seniority levels'), P164 ('one man company') and P162 "
        "('first-year PhD student') are mapped to 'Other' by the `mapping` dict — none match a "
        "Junior/Mid/Senior/Lead keyword.\n\n"
        "**Domain compares two independent groups via Mann-Whitney U**: participants naming "
        "IoT, Mobile, or an 'Embedded'-keyword free-text answer ('Energy-aware') vs. "
        "participants naming only other domains ('Other domains'). Participants who named "
        "domains from **both** groups are excluded to keep the two groups independent — "
        "**9–10 participants per scenario** (10 distinct individuals across the dataset: "
        "P2, P5, P39, P95, P112, P126, P128, P148, P155, P164 — e.g. P164 checked both "
        "'Mobile'/'IoT' and 'Infrastruktur'). Participants with no PQ3 answer at all are "
        "excluded as well (0 in this population — everyone in the Strategy 2 scenario samples "
        "answered PQ3)."
    )
    container.table(subgroup_df)


def eval_scenario_sensitivity_analysis(df, container):
    container.write(
        "##### Sensitivity Analysis: How does the strategy for selecting/resolving participant "
        "answers affect the correlation-scenario accuracy analysis (see above)?"
    )
    container.write(
        "**Strategy 1** — keep only participants who gave exactly one answer for the scenario. \n\n"
        "**Strategy 2** — additionally resolve participants with multiple *non-contradicting* "
        "answers to their most extreme selection (e.g. 'moderate positive' + 'perfekte positive' "
        "→ 'perfekte positive'); participants whose selections contradict each other (mixing "
        "positive/negative/no-correlation) are excluded \n\n"
        "**Strategy 3** — keep single-answer participants unchanged; for participants with "
        "multiple answers, randomly pick one of their given answers, repeated up to 30 times with "
        "deduplicated (non-repeating) answer combinations, capped at however many distinct "
        "combinations actually exist. Results are aggregated (p-value ranges and % significant) "
        "across all repetitions, using a fixed random seed for reproducibility. \n\n"
        "Filtering is applied independently per scenario, so a participant may be included for "
        "one scenario and excluded for another. The ternary score itself (0/1/2) is computed the "
        "same way regardless of strategy. The strategies only differ in which answers/decisions "
        "are included."
    )

    tmp_df, scale_5pt, scale_3pt, correct_answers = _prepare_correlation_df(df)

    scenario_labels = {
        'Scenario1': 'Scenario 1 (correct: perfekte positive)',
        'Scenario2': 'Scenario 2 (correct: moderate positive)',
        'Scenario3': 'Scenario 3 (correct: keine Korrelation)',
        'Scenario4': 'Scenario 4 (correct: moderate negative)',
    }

    _render_strategy2_pvalue_summary(tmp_df, correct_answers, container)
    _render_strategy2_subgroup_sizes(tmp_df, correct_answers, container)

    sc_tabs = container.tabs(list(scenario_labels.values()))
    for tab, scenario in zip(sc_tabs, scenario_labels):
        correct, scale = correct_answers[scenario]
        col1, col2, col3 = tab.columns(3)

        col1.write("**Strategy 1: Single answer only, removed participants with multiple answers**")
        strat1_df = _strategy_single_answer_only(tmp_df, scenario, correct, scale)
        p1_by_analysis = _render_sensitivity_strategy_results(strat1_df, scenario, col1, key_prefix=f"{scenario}_s1")

        col2.write("**Strategy 2: Remove Contradicting, Include Extrema**")
        strat2_df = _strategy_resolve_noncontradicting(tmp_df, scenario, correct, scale)
        p2_by_analysis = _render_sensitivity_strategy_results(strat2_df, scenario, col2, key_prefix=f"{scenario}_s2")

        col3.write("**Strategy 3: Random resolution of multi-answers (up to 30 unique repetitions)**")
        p3_by_analysis = _render_sensitivity_strategy3_summary(tmp_df, scenario, correct, scale, col3)

        _render_strategy_pvalue_comparison(p1_by_analysis, p2_by_analysis, p3_by_analysis, tab)


def prep_RE0(df, container):
    container.write("##### RE0: Welche Maßnahmen ergreifen Sie normalerweise wenn sich der Energieverbrauch der Software, die Sie entwickeln, erhöht?")
    category_mapping = {
        'RE0[SQ001]': 'Documenting the energy regression',
        'RE0[SQ002]': 'Analyzing the root-cause',
        'RE0[SQ003]': 'Fixing identified root-cause',
        'RE0[SQ004]': 'We ignore it',
        #'RE0[SQ005]': 'Ich weiß es nicht'
    }
    counts = {}
    for col, name in category_mapping.items():
        counts[name] = (df[col] == 'Ja').sum()

    other_counts = df['AE3[other]'].dropna().value_counts()
    # First, make DataFrame of main counts
    main_counts_df = pd.DataFrame(list(counts.items()), columns=['Category', 'Count'])

    # Now create the detailed Sonstiges DataFrame
    detailed_other_df = other_counts.reset_index()
    detailed_other_df.columns = ['Category', 'Count']

    full_counts_df = pd.concat([main_counts_df, detailed_other_df], ignore_index=True)
    full_counts_df_sorted = full_counts_df.sort_values('Count', ascending=False)

    custom_plt.plot_df_barplot(full_counts_df_sorted, container, 'Category', 
                               './app/figures/energy_consumption_assessment_measures.png', '', '')

def prep_RE1(df, container):
    container.write("##### RE1: Ist der Debugging Prozess von Energieproblemen anders als der Debugging Prozess für Performanceprobleme oder funktionale Fehler?")
    filtered_df = df[pd.notna(df['RE1'])]
    custom_plt.plot_countplot(filtered_df, container, 'RE1', './app/figures/energy_vs_perf_debugging.png', 'Frequency', 'Developers',
                   False, 1)

def prep_RE2(df, container):
    container.write(
        "##### RE2: Bitte beschreiben Sie Gemeinsamkeiten und Unterschiede beim Debuggen von Energieproblemen gegenüber anderen Fehlern, wie performance- und funktionalen Fehlern:"
    )

    question_column = 'RE2'
    filtered_df = df[pd.notna(df[question_column])]
    generate_open_answer_expander(filtered_df, container, question_column)


def prep_CO0(df, container):
    container.write(
        "##### CO0: Wenn Sie an Ihre vorherigen Antworten denken: Was sind die größten Probleme und Herausforderungen, "
        "die noch überwunden werden müssen, damit der Energieverbrauch von Software eine größere Rolle spielt?"
    )
    question_column = 'CO0'
    filtered_df = df[pd.notna(df[question_column])]
    generate_open_answer_expander(filtered_df, container, question_column)

def prep_CO1(df, container):
    container.write(
        "##### CO1: Was muss Ihrer Meinung nach getan werden, um die Herausforderungen des Energieverbrauchs zu bewältigen?"
    )
    question_column = 'CO1'
    filtered_df = df[pd.notna(df[question_column])]
    generate_open_answer_expander(filtered_df, container, question_column)

def prep_CQ3(df, container):
    container.write(
        "##### CQ3: Glauben Sie, dass es mehr Regulierungen (z. B. Gesetze) oder politische Anreize (z. B. spezielle Förderungen) "
        "braucht, um die Praxisrelevanz des Energieverbrauchs von Softwareprodukten zu erhöhen? Bitte erläutern Sie Ihre Sichtweise."
    )
    question_column = 'CQ3'
    filtered_df = df[pd.notna(df[question_column])]
    generate_open_answer_expander(filtered_df, container, question_column)


def prep_CQ0(df, container):
    container.write(
        "##### CQ0:  	Gibt es noch etwas, dass Sie uns bezüglich des Energieverbrauchs von Software oder allgemein zu dieser Umfrage mitteilen möchten?"
    )
    question_column = 'CQ0'
    filtered_df = df[pd.notna(df[question_column])]
    generate_open_answer_expander(filtered_df, container, question_column)

def prep_CQ1(df, container):
    container.write(
        "##### CQ1: Wir planen Interviews durchzuführen, um ein tieferes Verständnis für energiebewusste Softwareentwicklung "
        "in der Praxis zu gewinnen. Wenn Sie an einer Teilnahme interessiert sind, geben Sie bitte Ihre E-Mail-Adresse an. "
        "Wir senden Ihnen eine Einladung mit weiteren Informationen, sodass Sie entscheiden können, ob Sie teilnehmen möchten. "
        "Ihre E-Mail-Adresse:"
    )
    question_column = 'CQ1'
    filtered_df = df[pd.notna(df[question_column])]
    container.write(filtered_df[question_column].to_list())




def eval_filter_questions(df, container):
    df.loc[:, 'RQ4'] = df['RQ4'].str.strip()

    mask_relevant = df['RQ4'].str.startswith('Ja, der Energieverbrauch ist relevant', na=False)
    mask_not_relevant = df['RQ4'].str.startswith('Nein, der Energieverbrauch ist nicht relevant', na=False)
    mask_empty = df['RQ4'].isna() | (df['RQ4'] == '')
    df_relevant = df[mask_relevant]
    df_not_relevant = df[mask_not_relevant]
    df_empty = df[mask_empty]

    container.write("All: " + str(len(df)))

    container.write("F1 dropout: " + str(len(df_empty)))
    container.write("F1 relevant: " + str(len(df_relevant)))
    container.write("F1 not relevant: " + str(len(df_not_relevant)))
    container.write("")

    # Filter N2:
    mask_ja = df_not_relevant['FilterN2'].str.strip() == "Ja"
    mask_nein = df_not_relevant['FilterN2'].str.strip() == "Nein"
    mask_empty = df_not_relevant['FilterN2'].isna() | (df_not_relevant['FilterN2'].str.strip() == "")
    df_N2_yes = df_not_relevant[mask_ja]
    df_N2_no = df_not_relevant[mask_nein]
    df_N2_empty = df_not_relevant[mask_empty]

    container.write("N2 Yes, other factors relevant: " + str(len(df_N2_yes)))
    container.write("N2 No other factors Relevant: " + str(len(df_N2_no)))
    container.write("N2 Termination: " + str(len(df_N2_empty)))
    container.write("")


    # Filter Y2
    direct_cols = ["FilterY2[direct1]", "FilterY2[direct2]"]
    indirect_cols = [
        "FilterY2[indirect1]", "FilterY2[indirect2]", "FilterY2[indirect3]",
        "FilterY2[indirect4]", "FilterY2[indirect5]", "FilterY2[indirect6]",
        "FilterY2[indirect7]", "FilterY2[indirect8]"
    ]
    other_cols = ["FilterY2[other1]", "FilterY2[other2]", "FilterY2[other]"]
    has_ja_direct = df_relevant[direct_cols].eq("Ja").any(axis=1)
    has_ja_indirect = df_relevant[indirect_cols].eq("Ja").any(axis=1)
    has_ja_other = df_relevant[other_cols].apply(lambda row: row.astype(str).str.strip().ne("").any(), axis=1)
    
    set1_mask = has_ja_direct & ~has_ja_indirect
    set3_mask = has_ja_direct & has_ja_indirect
    set2_mask = ~has_ja_direct & has_ja_indirect

    set1 = df_relevant[set1_mask]
    set2 = df_relevant[set2_mask]
    set3 = df_relevant[set3_mask]

    container.write("Y2 only indirect measurements: " + str(len(set2)))
    container.write("Y2 direct and indirect measuremets: " + str(len(set3)))
    container.write("Y2 only direct measurements: " + str(len(set1)))

    container.write("Test set all direct: " + str(len(df_relevant[has_ja_direct])))
    container.write("Test set all indirect: " + str(len(df_relevant[has_ja_indirect])))
    container.write("Test set all other: " + str(len(df_relevant[has_ja_other])))


def extract_scenario_siniority_data(df, category_mapping):
    df.loc[:, "ec_relevance"] = pd.NA

    df.loc[
        df["RQ4"].str.startswith("Ja, der Energieverbrauch ist relevant", na=False),
        "ec_relevance"
    ] = "relevant"

    df.loc[
        df["RQ4"].str.startswith("Nein, der Energieverbrauch ist nicht relevant", na=False),
        "ec_relevance"
    ] = "nicht relevant"

    cols = list(category_mapping.keys())

    long = (
        df[cols + ["ec_relevance"]]
        .melt(id_vars="ec_relevance", value_vars=cols,
            var_name="scenario_col", value_name="answer")
    )

    long = long[long["answer"] == "Ja"].copy()
    long.loc[:, "Correlation"] = long["scenario_col"].map(category_mapping)

    counts = (
        long.groupby(["Correlation", "ec_relevance"])
            .size()
            .reset_index(name="Count")
    )
    table = (
        counts.pivot(index="Correlation",
                    columns="ec_relevance",
                    values="Count")
            .fillna(0)
            .astype(int)
    )

    return table


def extract_scenario_multiselection_data(df, category_mapping):
    cols = list(category_mapping.keys())
    selected = df[cols].eq("Ja")
    n_selected = selected.sum(axis=1)
    participant_mask = n_selected > 0   # participants who actually selected at least one correlation mode
    multi_mask = n_selected > 1

    multi_df = df.loc[multi_mask, ["id"]].copy()
    multi_df["n_selected"] = n_selected[multi_mask]
    multi_df["selected_correlations"] = selected.loc[multi_mask].apply(
        lambda row: ", ".join(category_mapping[c] for c in cols if row[c]),
        axis=1
    )

    n_participants = int(participant_mask.sum())   # participants who answered this scenario
    n_answers = int(selected.sum().sum())           # total "Ja" ticks across all categories (>= n_participants if multi-selected, < n_participants if some gave 0)

    # Distribution of how many options each participant selected (0 = "Nein" to everything).
    # sum(n_selected * n_participants_with_that_count) must equal n_answers, and
    # sum(n_participants_with_that_count) must equal n_participants.
    selection_counts = (
        n_selected[participant_mask]
        .value_counts()
        .sort_index()
        .rename_axis("n_selected")
        .reset_index(name="n_participants")
    )

    return multi_df.reset_index(drop=True), int(multi_mask.sum()), n_participants, n_answers, selection_counts


def report_scenario_multiselection(df, category_mapping, scenario_name, container):
    multi_df, n_multi, n_participants, n_answers, selection_counts = extract_scenario_multiselection_data(df, category_mapping)
    container.write(f"### {scenario_name}: Participants selecting multiple correlation modes")
    container.write(
        f"{n_participants} participants answered this scenario, giving {n_answers} answers in total "
        f"(consistency check against the sum of the table below). "
        f"{n_multi} of {n_participants} participants selected more than one correlation mode."
    )
    container.write("Breakdown by number of selections per participant:")
    container.write(selection_counts)
    if n_multi > 0:
        container.write(multi_df)
    container.write("")


def eval_scenario_multiselection(df, container):

    category_mapping_sc1 = {
        "Scenario1[SQ001]": "perfekte positive",
        "Scenario1[SQ002]": "moderate positive Korrelation",
        "Scenario1[SQ003]": "moderate negative Korrelation",
        "Scenario1[SQ004]": "perfekte negative Korrelation",
        "Scenario1[SQ005]": "keine Korrelation",
    }
    category_mapping_sc2 = {
        'Scenario2[SQ001]': 'perfekte positive',
        'Scenario2[SQ002]': 'moderate positive Korrelation',
        'Scenario2[SQ003]': 'moderate negative Korrelation',
        'Scenario2[SQ004]': 'perfekte negative Korrelation',
        'Scenario2[SQ005]': 'keine Korrelation'
    }
    category_mapping_sc3 = {
        'Scenario3[SQ001]': 'perfekte Korrelation',
        'Scenario3[SQ002]': 'moderate Korrelation',
        'Scenario3[SQ003]': 'keine Korrelation'
    }
    category_mapping_sc4 = {
        'Scenario4[SQ001]': 'perfekte positive',
        'Scenario4[SQ002]': 'moderate positive Korrelation',
        'Scenario4[SQ003]': 'moderate negative Korrelation',
        'Scenario4[SQ004]': 'perfekte negative Korrelation',
        'Scenario4[SQ005]': 'keine Korrelation'
    }

    report_scenario_multiselection(df, category_mapping_sc1, "Scenario 1", container)
    report_scenario_multiselection(df, category_mapping_sc2, "Scenario 2", container)
    report_scenario_multiselection(df, category_mapping_sc3, "Scenario 3", container)
    report_scenario_multiselection(df, category_mapping_sc4, "Scenario 4", container)

    return


def eval_scenario_seniority(df, container):

    category_mapping_sc1 = {
        "Scenario1[SQ001]": "perfekte positive",
        "Scenario1[SQ002]": "moderate positive Korrelation",
        "Scenario1[SQ003]": "moderate negative Korrelation",
        "Scenario1[SQ004]": "perfekte negative Korrelation",
        "Scenario1[SQ005]": "keine Korrelation",
    }
    category_mapping_sc2 = {
        'Scenario2[SQ001]': 'perfekte positive',
        'Scenario2[SQ002]': 'moderate positive Korrelation',
        'Scenario2[SQ003]': 'moderate negative Korrelation',
        'Scenario2[SQ004]': 'perfekte negative Korrelation',
        'Scenario2[SQ005]': 'keine Korrelation'
    }
    category_mapping_sc3 = {
        'Scenario3[SQ001]': 'perfekte Korrelation',
        'Scenario3[SQ002]': 'moderate Korrelation',
        'Scenario3[SQ003]': 'keine Korrelation'
    }
    category_mapping_sc4 = {
        'Scenario4[SQ001]': 'perfekte positive',
        'Scenario4[SQ002]': 'moderate positive Korrelation',
        'Scenario4[SQ003]': 'moderate negative Korrelation',
        'Scenario4[SQ004]': 'perfekte negative Korrelation',
        'Scenario4[SQ005]': 'keine Korrelation'
    }

    table_sc1 = extract_scenario_siniority_data(df, category_mapping_sc1)
    container.write("### Scenario 1: Seniority vs. Correlation beliefs")
    container.write(table_sc1)
    container.write("")
    table_sc2 = extract_scenario_siniority_data(df, category_mapping_sc2)
    container.write("### Scenario 2: Seniority vs. Correlation beliefs")
    container.write(table_sc2)
    container.write("")
    table_sc3 = extract_scenario_siniority_data(df, category_mapping_sc3)
    container.write("### Scenario 3: Seniority vs. Correlation beliefs")
    container.write(table_sc3)
    container.write("")
    table_sc4 = extract_scenario_siniority_data(df, category_mapping_sc4)
    container.write("### Scenario 4: Seniority vs. Correlation beliefs")
    container.write(table_sc4)

    return


def _participants_flow_counts(df):
    """All Sankey link values, computed from the survey data (nothing hardcoded) so the flow
    always adds up: every node's inflow equals its outflow, and 'Finishing Questions' equals the
    number of participants who reached the last survey page."""
    attended = df['RQ4'].notna()
    relevant = df['RQ4'].astype(str).str.startswith('Ja')
    not_relevant = df['RQ4'].astype(str).str.startswith('Nein')
    n2 = df['FilterN2'].astype(str).str.strip()
    finished = df['lastpage'] == df['lastpage'].max()

    direct_cols = [c for c in df.columns if c.startswith('FilterY2[direct')]
    indirect_cols = [c for c in df.columns if c.startswith('FilterY2[indirect')]
    y2_cols = [c for c in df.columns if c.startswith('FilterY2[')]
    has_direct = (df[direct_cols] == 'Ja').any(axis=1)
    has_indirect = (df[indirect_cols] == 'Ja').any(axis=1)
    y2_answered = df[y2_cols].notna().any(axis=1)

    c = {
        'all': len(df), 'stop': int((~attended).sum()), 'attended': int(attended.sum()),
        'relevant': int(relevant.sum()), 'not_relevant': int(not_relevant.sum()),
        'terminated': int((not_relevant & ~n2.isin(['Ja', 'Nein'])).sum()),
        'n2_yes': int((not_relevant & (n2 == 'Ja')).sum()),
        'n2_no': int((not_relevant & (n2 == 'Nein')).sum()),
        'y2_indirect': int((relevant & ~has_direct & has_indirect).sum()),
        'y2_both': int((relevant & has_direct & has_indirect).sum()),
        'y2_direct': int((relevant & has_direct & ~has_indirect).sum()),
        'y2_neither': int((relevant & y2_answered & ~has_direct & ~has_indirect).sum()),
        'y2_unanswered': int((relevant & ~y2_answered).sum()),
    }
    c['proxy'] = c['n2_yes'] + c['y2_indirect'] + c['y2_both']
    # Participants who completed the survey, by the branch they came from.
    c['fin_proxy'] = int((finished & ((not_relevant & (n2 == 'Ja')) | (relevant & has_indirect))).sum())
    c['fin_n2_no'] = int((finished & not_relevant & (n2 == 'Nein')).sum())
    c['fin_y2_direct'] = int((finished & relevant & has_direct & ~has_indirect).sum())
    c['fin_y2_neither'] = int((finished & relevant & y2_answered & ~has_direct & ~has_indirect).sum())
    return c


def plot_participants_flow(df, container):
    container.write(
        "### Flow of participants through the study."
    )

    container.write("")

    c = _participants_flow_counts(df)
    labels = ["All", "Immediate Stop", "Attended", "Is Relevant", "Not Relevant", "Terminated",
              # 0     1                 2           3              4               5
              "N2 Yes, Other Factors", "N2 No Other Factors", "Y2: Only indirect", "Y2: direct and indirect",
              # 6                       7                      8                    9
              "Y2: only direct", "Proxy", "Finishing Questions", "Y2: neither direct nor indirect",
              # 10                11       12                     13
              "Dropped out"]
              # 14
    links = [
        (0, 1, c['stop']), (0, 2, c['attended']),
        (2, 3, c['relevant']), (2, 4, c['not_relevant']),
        (4, 5, c['terminated']), (4, 6, c['n2_yes']), (4, 7, c['n2_no']),
        (3, 8, c['y2_indirect']), (3, 9, c['y2_both']), (3, 10, c['y2_direct']),
        (3, 13, c['y2_neither']), (3, 14, c['y2_unanswered']),
        (6, 11, c['n2_yes']), (8, 11, c['y2_indirect']), (9, 11, c['y2_both']),
        (11, 12, c['fin_proxy']), (11, 14, c['proxy'] - c['fin_proxy']),
        (7, 12, c['fin_n2_no']), (7, 14, c['n2_no'] - c['fin_n2_no']),
        (10, 12, c['fin_y2_direct']), (10, 14, c['y2_direct'] - c['fin_y2_direct']),
        (13, 12, c['fin_y2_neither']), (13, 14, c['y2_neither'] - c['fin_y2_neither']),
    ]
    links = [l for l in links if l[2] > 0]

    go_fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color="lightgray"  # Light background instead of blue
        ),
        link=dict(
            source=[l[0] for l in links],  # indices correspond to labels
            target=[l[1] for l in links],
            value=[l[2] for l in links]
        ))])

    go_fig.update_layout(title_text="Flow of Participants")
    container.plotly_chart(go_fig, use_container_width=False)
    go_fig.write_image("participants_flow.png", scale=3)

def generate_open_answer_expander(comments, container, question_code):

    with container:

        if show_category_graph:
                #st.write(f"data/category-graphs/{question_code}.drawio.png")
                st.image(f"./app/data/category-graphs/{question_code}.drawio.png")

        with st.expander(f"{question_code}: ({len(comments)} open answers)"):
            all_cats = categories_df.loc[categories_df["QID"] == question_code, "Combined Category"].unique().tolist()
            all_cats.sort()
            
            category = st.selectbox("Filter answers by categories:", ["show all [no filter]"] + all_cats, key="category-" + question_code)

            for _, row in comments.iterrows():
                cats = categories_df.loc[(categories_df["QID"] == question_code) & (categories_df["PID"] == row["id"]), "Combined Category"].tolist()
                if "all" in category or category in cats:
                    st.write("P" + str(row["id"]) +": *'" + row[question_code].replace("\n", " ").strip() + "'*" + "(:blue[Categories: " + ", ".join(cats) + "])")




mapping = {
    'Small team without seniority levels' : 'Other',
    'one man company' : 'Other',
    'Mid Level ' : 'Senior',
    'Mid' : 'Senior',
    'Manager' : 'Other',
    'Hobbyist' : 'Other', 
    'first-year PhD student' : 'Other',

    'Continiously in production via our own monitoring and fleet management solution.' : 'Continuously in production via our own monitoring and fleet management solution',

    'Forschung' : 'Research',
    'Software Engineering Research ' : 'Research',
    'Gamedev' : 'Game Development',
    'Games' : 'Game Development',
    'Game' : 'Game Development',
    'Game Industry' : 'Game Development',
    'GAMEDEV' : 'Game Development',
    'Games/Realtime' : 'Game Development',
    'academia' : 'Research',
    'Embedded Industrial Control' : 'Embedded',
    'desktop software for arts' : 'Desktop',
    'Industriemaschienen' : 'Industry',

    'Compilers' : 'Compiler',
    'Embedded Developer' : 'Embedded',
    'Embedded of varying sizes, PC, mobile' : 'Embedded',
    'Embedded Engineer' : 'Embedded',
    'It’s spelled Architecture. Should add embedded if you’re asking about energy consumption. ' : 'No answer',
    'Just developer (Not everything is web based...)' : 'Full-Stack',

    'fehlende Messverfahren' : 'Lack of tooling',
    'Management would be open at "acceptable cost", but even then it would be too expensive donw the chain.': 'Too expensive',
    'Zu teuer': 'Too expensive',
    "We design hardware, so while energy efficiency is hugely important for us, it doesn't matter at all in the design process.": 'Not relevant for our use case',
    'we can not measure energy consumption in any useful way.': 'Lack of tooling',
    'Faster performance trumped power usage.': 'Runtime performance is a proxy',
    'directly proportianal to efficiency (server are written in Go)': 'Runtime performance is a proxy',
    'Most of our software is already relatively efficient, so any savings would likely be modest': 'Not worth the effort',
    'Small installed base (a few hundred computers worldwide)':'Not worth the effort',
    'nobody cares':'Not worth the effort',

    'Vornehmlich Betrieb und DevOps': 'Operations and DevOps',
    'Automatische Skalierung': 'Operations and DevOps',
    'Verzicht auf unnötige Features und Daten': 'Source-code level',
    'Decision on overall system design.': 'Design and construction',


}


# Main
df = pd.read_csv(lime_survey_csv)
df = df.replace(mapping)

categories_df = pd.read_csv(card_sorting_categories_csv)
categories_df["Combined Category"] = categories_df.apply(
    lambda row: f"{row['General Category']} - {row['Specific Category']}"
    if pd.notna(row['General Category']) and row['General Category'] != ""
    else row['Specific Category'],
    axis=1
)



if page == "RQ 1: Relevance":
    st.header("RQ1: To what extent is energy consumption a relevant factor in software development decisions and actions?")
    prep_RQ1(df, st.container())
    prep_RQ2(df, st.container())
    prep_RQ4(df, st.container())
    prep_Filter_N2(df, st.container())

    custom_plt.plot_perceived_importance_likert('./app/figures/importance_distributions_likert_multiple.pdf')


elif page == "RQ 2: Assessment":
    st.header("RQ2: How is software energy consumption assessed by practitioners amid challenges of integrating it into development workflows?")
    prep_Filter_Y2(df, st.container())
    prep_AE2(df, st.container())
    prep_AE3(df, st.container())
    prep_AE4(df, st.container())
    prep_ECP0(df, st.container())
    prep_EnergyProxy01_02(df, st.container())


elif page == "RQ 3: Mitigation Strategies":
    st.header("RQ3: What strategies and development practices are used to mitigate software energy consumption?")
    prep_EPT0(df, st.container())
    prep_EPT1(df, st.container())
    prep_RE0(df, st.container())
    prep_RE1(df, st.container())
    prep_RE2(df, st.container())

    
elif page == "RQ 4: Proxies Old":
    st.header("RQ4: What mental models do practitioners use to relate software energy consumption to other metrics like runtime performance?")
    prep_EnergyProxy03(df, st.container())
    prep_ProxyMerge01(df, st.container())
    prep_Scenario1(df, st.container())
    prep_Scenario2(df, st.container())
    prep_Scenario3(df, st.container())
    prep_Scenario4(df, st.container())

    custom_plt.plot_proxy_suitability_likert('./app/figures/proxy_metric_suitability_likert.pdf')
    scenario_likert, scenario_likert_by_relevance = _strategy2_scenario_likert_data(df)
    custom_plt.plot_scenario_likert('./app/figures/scenario_distributions_likert.pdf', scenario_likert)
    custom_plt.plot_scenario_likert_relevance('./app/figures/scenario_distributions_likert_relevance.pdf', scenario_likert_by_relevance)

elif page == "RQ 4: Proxy Analysis Revisions":
    st.header("RQ4: Proxy Analysis Revisions")

    with st.expander("Multi-Scenario-Selection analysis"):
        eval_scenario_multiselection(df.copy(), st.container())

    with st.expander("Correlation Scenario Sensitivity Analysis", expanded=True):
        eval_scenario_sensitivity_analysis(df.copy(), st.container())

# Hidden from the sidebar menu (see the commented-out entry in the radio list above).
elif page == "Revision: Extended Proxys Evaluation":
    st.header("RQ4: What mental models do practitioners use to relate software energy consumption to other metrics like runtime performance (Revision Extension)?")
    correlation_analysis(df, st.container())

    with st.expander("Correlation Scenario Sensitivity Analysis", expanded=True):
        eval_scenario_sensitivity_analysis(df, st.container())


elif page == "RQ 5: Incentives":
    st.header("RQ5: What strategies, resources, or incentives could support practitioners in better managing software energy use?")
    prep_CO0(df, st.container())
    prep_CO1(df, st.container())
    prep_CQ3(df, st.container())


elif page == "Personal Background":
    st.header("Personal Background")
    # Same cohort for all four questions: participants who passed the initial screening (RQ4
    # answered), i.e. the N=134 "Attended" node of the Sankey flow.
    background_df = df[df['RQ4'].notna()].copy()
    st.write(f"Participants who passed the initial screening: N={len(background_df)}")
    prep_PQ1(background_df, st.container())
    prep_PQ2(background_df, st.container())
    prep_PQ3(background_df, st.container())
    prep_PQ4(background_df, st.container())


#elif page == "Fundamental Disagreements":
#    st.header("Fundamental Disagreements")
#    st.write()


elif page == "Data Exploration":
    st.header("Data Exploration")

    with st.expander("Survey Data"):
        st.write(df)

    with st.expander("Card Sorting Categories"):
        st.write(categories_df)

    with st.expander("Filter Question Evaluation"):
        eval_filter_questions(df.copy(), st.container())
    
    with st.expander("Seniority Influence on Correlation Scenarios"):
        eval_scenario_seniority(df.copy(), st.container())

    
    plot_participants_flow(df, st.container())

    prep_CQ0(df, st.container())
    prep_CQ1(df, st.container())
