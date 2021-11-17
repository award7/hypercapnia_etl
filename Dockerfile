FROM apache/airflow
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir scipy plotly