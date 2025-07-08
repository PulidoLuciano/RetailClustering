-- init.sql
CREATE DATABASE mlflow_db;
CREATE USER mlflow_user WITH ENCRYPTED PASSWORD 'mlflow_pass';
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;
