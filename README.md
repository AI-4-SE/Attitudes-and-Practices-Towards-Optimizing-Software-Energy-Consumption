# Developer Attitudes and Practices Towards Optimizing Software Energy Consumption

This repository accompanies our ACM Transactions on Software Engineering and Methodology (TOSEM) submission **“Developer Attitudes and Practices Towards Optimizing Software Energy Consumption.”** It provides the artifact package for inspecting the study material and reproducing the analyses and figures reported in the paper.

The study is based on an online survey implemented with **LimeSurvey** and conducted with software developers recruited from green software engineering–related communities. This repository includes:
- the survey data export,
- the results of the card-sorting procedure, and
- the analysis scripts used to generate the plots and figures.

## Data

All data files are located in [`app/data`](./app/data):

- `results-survey762399.csv` — LimeSurvey export (tabular responses).
- `Card-Sorting.csv` — Overview and results of the card-sorting procedure.

## Reproducing the Analysis

All scripts required to re-run the evaluation are located in [`app`](./app). Results can be explored via a **Streamlit** application (recommended), either locally or through Docker.

### Option A: Run locally (Python)

1. Install dependencies:
```bash
pip install -r app/requirements.txt
```
2. Start the Streamlit app:
```bash
streamlit run app/app.py
```
3. Open the UI (default):   
- http://localhost:8501


### Option B: Run with Docker
1. Build and run the container:
```bash
cd app && docker compose up --build
```
or
```bash
docker build --network=host -t software-energy-consumption-evaluation app/
docker run -w / -p 127.0.0.1:8501:8501 software-energy-consumption-evaluation \
  streamlit run /app/app.py --server.port=8501
```
2. Open the UI:
- http://localhost:8501


## Figures

All generated figures are stored in [figures](./app/figures).
