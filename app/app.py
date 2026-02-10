import pandas as pd

import streamlit as st
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import json
import itertools
import logging
for name, l in logging.root.manager.loggerDict.items():
    if "streamlit" in name:
        l.disabled = True



# Global Variables
# /home/mweber/projects/energy conscious software engineering
lime_survey_csv = './app/data/results-survey762399.csv'
card_sorting_categories_csv = './app/data/Auswertung-Card-Sorting.csv'



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
    "RQ 4: Proxys", 
    "RQ 5: Incentives", 
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

    mask = df['PQ2'] == 'Sonstiges'
    df.loc[mask, 'PQ2'] = df.loc[mask].apply(
        lambda row: f"{row['PQ2[other]']}" if pd.notna(row['PQ2[other]']) else row['PQ2'], axis=1
    )
    df['PQ2'] = df['PQ2'].replace("we don't have a Junior/Senior split, but it would be Senior probably ", "Senior")

    custom_plt.plot_countplot(df, container, 'PQ2', './app/figures/seniority.png', 'Developers', 'Seniority')

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
    pq4_full_counts_df['Category'] = pq4_full_counts_df['Category'].str.strip().str.title()
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

    df['RQ4'] = df['RQ4'].replace("Nein, der Energieverbrauch ist nicht relevantfür in unseren Projekten, obwohl wir uns auf andere Eigenschaften (wie Laufzeit oder Cloud-Kosten) konzentrieren, die wiederum den Energieverbrauch beeinflussen können.", "Not relevant")
    df['RQ4'] = df['RQ4'].replace("Ja, der Energieverbrauch ist relevant in unseren Projekten und wir bewerten den Energieverbrauch unserer Software (z. B. über Energiemessungen, Laufzeitleistung, CPU-Auslastung, Kundenfeedback, Bauchgefühl oder anderes).", "Is relevant")

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
    filtered_df = df[pd.notna(df['AE4'])]

    filtered_df['AE4'] = filtered_df['AE4'].replace("Ja", "Yes")
    filtered_df['AE4'] = filtered_df['AE4'].replace("Nein", "No")
    filtered_df['AE4'] = filtered_df['AE4'].replace("Keine Antwort", "No response")

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

def prep_Scenario1(df, container):
    container.write(
        "##### Scenario1: Wie geeignet sind Ihrer Meinung nach die folgenden Metriken, als Indikator für den Energieverbrauch von Software?"
    )

    category_mapping = {
        'Scenario1[SQ001]': 'perfekte positive',
        'Scenario1[SQ002]': 'moderate positive Korrelation',
        'Scenario1[SQ003]': 'moderate negative Korrelation',
        'Scenario1[SQ004]': 'perfekte negative Korrelation',
        'Scenario1[SQ005]': 'keine Korrelation'
    }

    # df['RQ4']

    counts = {}
    for col, name in category_mapping.items():
        counts[name] = (df[col] == 'Ja').sum()

    # First, make DataFrame of main counts
    main_counts_df = pd.DataFrame(list(counts.items()), columns=['Category', 'Count'])
    custom_plt.plot_df_barplot(main_counts_df, container, 'Category', './app/figures/energy_correlation_scenario1.png',
                    'Count', 'Korrelation')

def prep_Scenario2(df, container):
    container.write(
        "##### Scenario2: Wie geeignet sind Ihrer Meinung nach die folgenden Metriken, als Indikator für den Energieverbrauch von Software?"
    )
    category_mapping = {
        'Scenario2[SQ001]': 'perfekte positive',
        'Scenario2[SQ002]': 'moderate positive Korrelation',
        'Scenario2[SQ003]': 'moderate negative Korrelation',
        'Scenario2[SQ004]': 'perfekte negative Korrelation',
        'Scenario2[SQ005]': 'keine Korrelation'
    }
    counts = {}
    for col, name in category_mapping.items():
        counts[name] = (df[col] == 'Ja').sum()

    # First, make DataFrame of main counts
    main_counts_df = pd.DataFrame(list(counts.items()), columns=['Category', 'Count'])
    custom_plt.plot_df_barplot(main_counts_df, container, 'Category', './app/figures/energy_correlation_scenario2.png',
                    'Count', 'Korrelation')

def prep_Scenario3(df, container):
    container.write(
        "##### Scenario3: Wie geeignet sind Ihrer Meinung nach die folgenden Metriken, als Indikator für den Energieverbrauch von Software?"
    )
    category_mapping = {
        'Scenario3[SQ001]': 'perfekte Korrelation',
        'Scenario3[SQ002]': 'moderate Korrelation',
        'Scenario3[SQ003]': 'keine Korrelation'
    }
    counts = {}
    for col, name in category_mapping.items():
        counts[name] = (df[col] == 'Ja').sum()

    # First, make DataFrame of main counts
    main_counts_df = pd.DataFrame(list(counts.items()), columns=['Category', 'Count'])
    custom_plt.plot_df_barplot(main_counts_df, container, 'Category', './app/figures/energy_correlation_scenario3.png',
                    'Count', 'Korrelation')

def prep_Scenario4(df, container):
    container.write(
        "##### Scenario4: Wie geeignet sind Ihrer Meinung nach die folgenden Metriken, als Indikator für den Energieverbrauch von Software?"
    )
    category_mapping = {
        'Scenario4[SQ001]': 'perfekte positive',
        'Scenario4[SQ002]': 'moderate positive Korrelation',
        'Scenario4[SQ003]': 'moderate negative Korrelation',
        'Scenario4[SQ004]': 'perfekte negative Korrelation',
        'Scenario4[SQ005]': 'keine Korrelation'
    }
    counts = {}
    for col, name in category_mapping.items():
        counts[name] = (df[col] == 'Ja').sum()

    # First, make DataFrame of main counts
    main_counts_df = pd.DataFrame(list(counts.items()), columns=['Category', 'Count'])
    custom_plt.plot_df_barplot(main_counts_df, container, 'Category', './app/figures/energy_correlation_scenario4.png',
                    'Count', 'Korrelation')


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
    df['RQ4'] = df['RQ4'].str.strip()

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
    df["ec_relevance"] = pd.NA

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
    long["Correlation"] = long["scenario_col"].map(category_mapping)

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


def eval_scenario_siniority(df, container):

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


def plot_participants_flow(df, container):
    container.write(
        "### Flow of participants through the study."
    )

    container.write("")

    go_fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["All", "Immediate Stop", "Attended", "Is Relevant", "Not Relevant", "Terminated",
                   # 0     1                 2           3              4               5
                   "N2 Yes, Other Factors", "N2 No Other Factors", "Y2: Only indirect", "Y2: direct and indirect",
                   # 6                       7                      8                    9
                   "Y2: only direct", "Proxy", "Finishing Questions"],
                   # 10                11       12
            color="lightgray"  # Light background instead of blue
        ),
        link=dict(
            source=[0, 0,   2,  2,  4, 4,  4,  6,  7,  3,  3,  8,  3,  9,  11],  # indices correspond to labels, eg A1, A2, A1, B1, ...
            target=[1, 2,   3,  4,  5, 6,  7,  11, 12, 10, 8,  11, 9,  11, 12],
            value=[90, 134, 45, 89, 9, 58, 22, 58, 22, 0,  14, 14, 19, 19, 91]
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
    'Small team without seniority levels' : 'Sole',
    'one man company' : 'Sole',
    'Mid Level ' : 'Senior',
    'Mid' : 'Senior',
    'Manager' : 'Other',
    'Hobbyist' : 'Other', 
    'first-year PhD student' : 'Other',

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
    'Faster performance trumped power usage.': 'Runtime performance is proxy',
    'directly proportianal to efficiency (server are written in Go)': 'Runtime performance is proxy',
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

    
elif page == "RQ 4: Proxys":
    st.header("RQ4: What mental models do practitioners use to relate software energy consumption to other metrics like runtime performance?")
    prep_EnergyProxy03(df, st.container())
    prep_ProxyMerge01(df, st.container())
    prep_Scenario1(df, st.container())
    prep_Scenario2(df, st.container())
    prep_Scenario3(df, st.container())
    prep_Scenario4(df, st.container())

    custom_plt.plot_proxy_suitability_likert('./app/figures/proxy_metric_suitability_likert.pdf')
    custom_plt.plot_scenario_likert('./app/figures/scenario_distributions_likert.pdf')
    custom_plt.plot_scenario_likert_relevance('./app/figures/scenario_distributions_likert_relevance.pdf')


elif page == "RQ 5: Incentives":
    st.header("RQ5: What strategies, resources, or incentives could support practitioners in better managing software energy use?")
    prep_CO0(df, st.container())
    prep_CO1(df, st.container())
    prep_CQ3(df, st.container())


elif page == "Personal Background":
    st.header("Personal Background")
    prep_PQ1(df, st.container())
    prep_PQ2(df, st.container())
    prep_PQ3(df, st.container())
    prep_PQ4(df, st.container())


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
    
    with st.expander("Seniority Influence on Correlation Scenarios", expanded=True):
        eval_scenario_siniority(df.copy(), st.container())
    
    plot_participants_flow(df, st.container())

    prep_CQ0(df, st.container())
    prep_CQ1(df, st.container())

