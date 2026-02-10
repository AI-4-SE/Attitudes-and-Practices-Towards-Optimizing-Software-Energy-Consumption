FROM python:3.12-slim-bookworm

WORKDIR /app

COPY . .

RUN ls
RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run", "/app/app.py", "–server.port=8501", "–server.address=0.0.0.0", "--browser.gatherUsageStats", "false",  "--server.fileWatcherType", "none"]