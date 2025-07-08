from dagster import Definitions, load_assets_from_modules
from dagstermill import ConfigurableLocalOutputNotebookIOManager
from dagster_mlflow import mlflow_tracking

from RetailClustering import assets  # noqa: TID252

all_assets = load_assets_from_modules([assets])

defs = Definitions(
    assets=all_assets,
    resources={
        "output_notebook_io_manager": ConfigurableLocalOutputNotebookIOManager(),
        "mlflow_kmeans": mlflow_tracking.configured({
            "experiment_name": "kmeans_clustering",
            "mlflow_tracking_uri": "http://localhost:5000",
        })
    }
)
