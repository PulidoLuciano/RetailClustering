# Dockerfile
FROM python:3.10-slim

# Instala MLflow y PostgreSQL driver
RUN pip install mlflow psycopg2-binary

EXPOSE 5000

CMD mlflow server \
    --backend-store-uri postgresql://mlflow_user:mlflow_pass@postgres:5432/mlflow_db \
    --host 0.0.0.0 \
    --port 5000 \
