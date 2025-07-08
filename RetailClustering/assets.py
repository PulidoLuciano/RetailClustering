import dagster as dg
import pandas as pd
from .utils import delete_cancelled_orders
from os import path
from dagstermill import define_dagstermill_asset

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