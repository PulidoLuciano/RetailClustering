import dagster as dg
import pandas as pd
from .utils import delete_cancelled_orders, cluster_data, plot_dendrogram
from dagstermill import define_dagstermill_asset
from os import path
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
    from .utils import winsorize_by_percentile
    from sklearn.preprocessing import RobustScaler
    winsorized_df = winsorize_by_percentile(rfm_data, lower_percentile=10, upper_percentile=90)
    
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(winsorized_df)
    scaled_df = pd.DataFrame(scaled_data, columns=winsorized_df.columns)
    return scaled_df

    return scaled_df

@dg.asset(
    dagster_type=pd.DataFrame,
    description="Clustering the RFM data with KMeans",
    group_name="clustering",
    required_resource_keys={"mlflow_kmeans"},
)
def clustered_kmeans_data(context: dg.AssetExecutionContext, scaled_rfm_data: pd.DataFrame) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    mlflow = context.resources.mlflow_kmeans
    N_CLUSTERS = 4
    RUN_NAME = "only_rfm"

    model = KMeans(n_clusters=N_CLUSTERS)
    results_df = cluster_data(scaled_rfm_data, model, RUN_NAME, mlflow)
    mlflow.log_metrics({"inertia": model.inertia_})
    mlflow.end_run()

    return results_df

@dg.asset(
    dagster_type=pd.DataFrame,
    description="Clustering the RFM data with DBSCAN",
    group_name="clustering",
    required_resource_keys={"mlflow_dbscan"},
)
def clustered_dbscan_data(context: dg.AssetExecutionContext, scaled_rfm_data: pd.DataFrame) -> pd.DataFrame:
    from sklearn.cluster import DBSCAN
    mlflow = context.resources.mlflow_dbscan
    RUN_NAME = "only_rfm"

    model = DBSCAN(eps=0.25, min_samples=20)
    results_df = cluster_data(scaled_rfm_data, model, RUN_NAME, mlflow)
    mlflow.end_run()
    return results_df

@dg.asset(
    dagster_type=pd.DataFrame,
    description="Clustering the RFM data with Agglomerative Clustering",
    group_name="clustering",
    required_resource_keys={"mlflow_agglomerative"},
)
def clustered_agglomerative_data(context: dg.AssetExecutionContext, scaled_rfm_data: pd.DataFrame) -> pd.DataFrame:
    from sklearn.cluster import AgglomerativeClustering
    mlflow = context.resources.mlflow_agglomerative
    RUN_NAME = "only_rfm"

    model = AgglomerativeClustering(n_clusters=4, linkage="ward")
    results_df = cluster_data(scaled_rfm_data, model, RUN_NAME, mlflow)
    #plot_dendrogram(model, mlflow)
    mlflow.end_run()

    return results_df

@dg.asset(
    dagster_type=pd.DataFrame,
    description="Clustering the RFM data with Gaussian Mixture",
    group_name="clustering",
    required_resource_keys={"mlflow_gaussian_mixture"},
)
def clustered_gaussian_mixture_data(context: dg.AssetExecutionContext, scaled_rfm_data: pd.DataFrame) -> pd.DataFrame:
    from sklearn.mixture import GaussianMixture
    mlflow = context.resources.mlflow_gaussian_mixture
    RUN_NAME = "only_rfm"
    N_CLUSTERS = 4

    model = GaussianMixture(n_components=N_CLUSTERS)
    results_df = cluster_data(scaled_rfm_data, model, RUN_NAME, mlflow)
    mlflow.end_run()

    return results_df