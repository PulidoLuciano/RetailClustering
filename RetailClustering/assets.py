import dagster as dg
import pandas as pd
from .utils import delete_cancelled_orders, get_pca
from os import path
from dagstermill import define_dagstermill_asset
from sklearn.metrics import silhouette_score
RANDOM_STATE = 42

@dg.asset(
    dagster_type=pd.DataFrame,
    description="Raw data from the online retail dataset",
    group_name="data_ingestion",
)
def raw_data():
    return pd.read_excel(path.join(path.dirname(__file__), '../data/raw_online_retail.xlsx'))


first_eda_nb = define_dagstermill_asset(
    name="first_eda_nb",
    notebook_path=dg.file_relative_path(__file__, "./notebooks/first_eda.ipynb"),
    description="Explanation and visualization of the first cleaning of the data",
    group_name="preprocessing",
    ins={"raw_data": dg.AssetIn(key=dg.AssetKey("raw_data"))},
)

@dg.asset(
    dagster_type=pd.DataFrame,
    description="Cleaned data from the online retail dataset",
    group_name="preprocessing",
)
def cleaned_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    retail_df = raw_data[raw_data['CustomerID'].notnull()].copy()
    retail_df = retail_df[retail_df['Quantity'] > 0]
    retail_df = delete_cancelled_orders(retail_df)
    retail_df = retail_df[retail_df['UnitPrice'] > 0]
    retail_df = retail_df.drop_duplicates()
    retail_df['StockCode'] = retail_df['StockCode'].astype(str)
    retail_df = retail_df[~retail_df['StockCode'].str.contains('^[a-zA-Z]',regex=True)] 
    return retail_df

@dg.asset(
    dagster_type=pd.DataFrame,
    description="Data with the total price and right types",
    group_name="preprocessing",
)
def preprocessed_data(cleaned_data: pd.DataFrame) -> pd.DataFrame:
    cleaned_data['TotalPrice'] = cleaned_data['Quantity'] * cleaned_data['UnitPrice']
    cleaned_data['InvoiceDate'] = pd.to_datetime(cleaned_data['InvoiceDate'])
    cleaned_data['CustomerID'] = cleaned_data['CustomerID'].astype(int)
    cleaned_data['InvoiceNo'] = cleaned_data['InvoiceNo'].astype(int)
    return cleaned_data

rfm_definitions_nb = define_dagstermill_asset(
    name="rfm_definitions_nb",
    notebook_path=dg.file_relative_path(__file__, "./notebooks/rfm_definitions.ipynb"),
    description="Definition of the RFM features and their transformations",
    group_name="preprocessing",
    ins={"preprocessed_data": dg.AssetIn(key=dg.AssetKey("preprocessed_data"))},
)

@dg.asset(
    dagster_type=pd.DataFrame,
    description="Data with the RFM features",
    group_name="preprocessing",
)
def rfm_data(preprocessed_data: pd.DataFrame) -> pd.DataFrame:
    # Recency
    fecha_referencia = preprocessed_data['InvoiceDate'].max() + pd.Timedelta(days=1)
    recency_df = preprocessed_data.groupby('CustomerID')['InvoiceDate'].max().reset_index()
    recency_df['Recency'] = (fecha_referencia - recency_df['InvoiceDate']).dt.days

    # Frequency
    frequency_df = preprocessed_data.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
    frequency_df.rename(columns={'InvoiceNo': 'Frequency'}, inplace=True)

    #Monetary
    monetary_df = preprocessed_data.groupby('CustomerID')['TotalPrice'].sum().reset_index()
    monetary_df.rename(columns={'TotalPrice': 'Monetary'}, inplace=True)

    #Merge
    rfm_df = recency_df.merge(frequency_df, on='CustomerID')
    rfm_df = rfm_df.merge(monetary_df, on='CustomerID')
    rfm_df.drop(columns=['InvoiceDate'], inplace=True)
    rfm_df = rfm_df.set_index('CustomerID')

    return rfm_df

@dg.asset(
    dagster_type=pd.DataFrame,
    description="Scaled and transformed RFM data for clustering",
    group_name="preprocessing",
)
def scaled_rfm_data(rfm_data: pd.DataFrame) -> pd.DataFrame:
    #from sklearn.preprocessing import StandardScaler
    #scaler = StandardScaler()
    #scaled_data = scaler.fit_transform(rfm_data[['Recency', 'Frequency', 'Monetary']])
    #scaled_data = pd.DataFrame(scaled_data, columns=['Recency', 'Frequency', 'Monetary'])
    #return scaled_data
    return rfm_data

@dg.asset(
    dagster_type=pd.DataFrame,
    description="Clustering the RFM data with KMeans",
    group_name="clustering",
    required_resource_keys={"mlflow_kmeans"},
)
def clustered_kmeans_data(context: dg.AssetExecutionContext, scaled_rfm_data: pd.DataFrame) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    
    N_CLUSTERS = 5
    PCA_components = 2
    RUN_NAME = "only_rfm"
    
    mlflow = context.resources.mlflow_kmeans
    mlflow.set_tag("mlflow.runName", RUN_NAME)
    mlflow.log_params({"random_state": RANDOM_STATE, "PCA_components": PCA_components})
    mlflow.log_params({"scaler": "-"})

    mlflow.autolog()
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    scaled_rfm_data['Cluster'] = kmeans.fit_predict(scaled_rfm_data)
    print(scaled_rfm_data.head(5))

    pca_fig, sum_explained_variance = get_pca(scaled_rfm_data, PCA_components)
    mlflow.log_metrics({"pca_explained_variance": sum_explained_variance})
    pca_fig.savefig("cache/pca_fig.png")
    mlflow.log_artifact("cache/pca_fig.png")

    mlflow.log_metrics({"inertia": kmeans.inertia_})
    mlflow.log_metrics({"silhouette_score": silhouette_score(scaled_rfm_data[['Recency', 'Frequency', 'Monetary']], scaled_rfm_data['Cluster'])})
    scaled_rfm_data.to_csv("cache/kmeans_model.csv")
    mlflow.log_artifact("cache/kmeans_model.csv")
    mlflow.sklearn.log_model(kmeans, "kmeans_model")

    mlflow.end_run()
    return scaled_rfm_data



