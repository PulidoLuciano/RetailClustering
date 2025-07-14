import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
import seaborn as sns
import numpy as np
from os import path, makedirs
from scipy.cluster.hierarchy import dendrogram
import math

def get_devolutions(df: pd.DataFrame) -> pd.DataFrame:
    devoluciones_df = df[df['Quantity'] < 0]
    return devoluciones_df

def delete_cancelled_orders(df: pd.DataFrame) -> pd.DataFrame:
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    devoluciones_df = get_devolutions(df)
    devoluciones_df['InvoiceDate'] = pd.to_datetime(devoluciones_df['InvoiceDate'])
    devoluciones_df['Quantity'] = -devoluciones_df['Quantity']

    merged = pd.merge(
        devoluciones_df,
        df,
        on=['CustomerID', 'StockCode', 'Quantity'],
        suffixes=('_dev', '_comp')
    )

    # Filtramos solo donde la compra fue antes que la devolución
    merged = merged[merged['InvoiceDate_comp'] < merged['InvoiceDate_dev']]

    # Calculamos diferencia de tiempo en horas
    merged['Diferencia_Horas'] = (merged['InvoiceDate_dev'] - merged['InvoiceDate_comp']).dt.total_seconds() / 3600

    # Ordenamos para tomar la compra más reciente antes de la devolución
    merged = merged.sort_values(by=['InvoiceNo_dev', 'Diferencia_Horas'])

    # Para cada devolución, nos quedamos con la compra más cercana
    matched = merged.groupby('InvoiceNo_dev').first().reset_index()

    retail_df = matched[[
        'CustomerID',
        'StockCode',
        'Description_dev',
        'InvoiceNo_comp', 'InvoiceDate_comp',
        'InvoiceNo_dev', 'InvoiceDate_dev',
        'Quantity',
        'Diferencia_Horas'
    ]]

    # Renombramos para mayor claridad
    retail_df.columns = [
        'CustomerID',
        'StockCode',
        'Description_dev',
        'InvoiceNo_Compra', 'Fecha_Compra',
        'InvoiceNo_Devolucion', 'Fecha_Devolucion',
        'Cantidad_Devuelta',
        'Diferencia_Horas'
    ]

    delete_devolutions_df = retail_df[retail_df['Diferencia_Horas'] < 72]
    # Creamos un set de tuplas con las combinaciones exactas a eliminar
    productos_devueltos = set(zip(delete_devolutions_df['InvoiceNo_Compra'], delete_devolutions_df['StockCode']))

    # Filtramos las filas que no están en productos_devueltos
    return df[~df.apply(lambda row: (row['InvoiceNo'], row['StockCode']) in productos_devueltos, axis=1)]

def get_pca(data: pd.DataFrame, n_components: int):
    """
    Toma los datos y los reduce a n_components componentes principales.

    Parámetros:
    - df: DataFrame con columnas para PC1, PC2 y la columna de clusters.
    - n_components: número de componentes principales.

    Retorna:
    - fig: objeto matplotlib.figure.Figure
    - sum_explained_variance: suma de los explained_variance_ratio_
    """
    from sklearn.decomposition import PCA
    df = data.copy()
    df.drop(columns=['Cluster'], inplace=True)
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(df)
    pca_results = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(n_components)], index=data.index)
    pca_results['Cluster'] = data['Cluster']
    print(pca_results.head(5))
    fig = plot_pca_clusters_figure(pca_results, pca.explained_variance_ratio_)
    sum_explained_variance = sum(pca.explained_variance_ratio_)
    return fig, sum_explained_variance

def plot_pca_clusters_figure(pca_df: pd.DataFrame, explained_variance: list, x_col="PC1", y_col="PC2", cluster_col="Cluster") -> Figure:
    """
    Genera una figura de Matplotlib con los clusters visualizados en el espacio PCA.

    Parámetros:
    - pca_df: DataFrame con columnas para PC1, PC2 y la columna de clusters.
    - x_col, y_col: nombres de las columnas para los ejes x e y.
    - cluster_col: nombre de la columna que contiene la asignación de clusters.

    Retorna:
    - fig: objeto matplotlib.figure.Figure
    """
    # Crear figura y ejes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Graficar cada cluster
    for cluster in sorted(pca_df[cluster_col].unique()):
        subset = pca_df[pca_df[cluster_col] == cluster]
        ax.scatter(subset[x_col], subset[y_col], label=f"Cluster {cluster}", alpha=0.6)

    # Configurar ejes y leyenda
    ax.set_title("Clusters visualizados en espacio PCA")
    ax.set_xlabel(f'{x_col} ({explained_variance[0]})')
    ax.set_ylabel(f'{y_col} ({explained_variance[1]})')
    ax.legend()
    ax.grid(True)

    return fig

def plot_silhouette(X, labels):
    silhouette_vals = silhouette_samples(X, labels)
    silhouette_avg = silhouette_score(X, labels)
    n_clusters = len(set(labels))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = silhouette_vals[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = sns.color_palette("hsv", n_clusters)[i]
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7
        )
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_title("Silhouette Plot")
    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])
    
    return fig, silhouette_avg

def plot_dendrogram(model, mlflow, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    plt.savefig("cache/dendrogram.png")
    mlflow.log_artifact("cache/dendrogram.png")

def winsorize_by_percentile(data, lower_percentile=5, upper_percentile=95):
    """
    Aplica winsorización a un DataFrame o a una Serie de Pandas.

    Los valores por debajo del percentil inferior se reemplazarán por el valor
    del percentil inferior. Los valores por encima del percentil superior se
    reemplazarán por el valor del percentil superior.

    Parámetros:
    -----------
    data : pd.DataFrame o pd.Series
        El conjunto de datos al que se aplicará la winsorización.
        Si es un DataFrame, la winsorización se aplica columna por columna.
        Si es una Serie, se aplica a la Serie.
    lower_percentile : int o float, opcional (default=5)
        El percentil inferior (e.g., 5 para el 5to percentil).
        Debe estar entre 0 y 100.
    upper_percentile : int o float, opcional (default=95)
        El percentil superior (e.g., 95 para el 95to percentil).
        Debe estar entre 0 y 100.

    Retorna:
    --------
    pd.DataFrame o pd.Series
        Los datos con la winsorización aplicada.
    """

    if not (0 <= lower_percentile < upper_percentile <= 100):
        raise ValueError("Los percentiles deben estar entre 0 y 100, "
                         "y lower_percentile debe ser menor que upper_percentile.")

    # Convertir a DataFrame si la entrada es una Serie para manejar ambos casos uniformemente
    if isinstance(data, pd.Series):
        is_series = True
        df = data.to_frame()
    elif isinstance(data, pd.DataFrame):
        is_series = False
        df = data.copy() # Trabajar con una copia para no modificar el DataFrame original
    else:
        raise TypeError("La entrada debe ser un pd.DataFrame o pd.Series.")

    winsorized_df = pd.DataFrame(index=df.index, columns=df.columns)

    for column in df.columns:
        # Asegurarse de que la columna sea numérica
        if pd.api.types.is_numeric_dtype(df[column]):
            lower_bound = np.percentile(df[column].dropna(), lower_percentile)
            upper_bound = np.percentile(df[column].dropna(), upper_percentile)

            winsorized_col = df[column].clip(lower=lower_bound, upper=upper_bound)
            winsorized_df[column] = winsorized_col
        else:
            # Si no es numérica, simplemente copiar la columna
            winsorized_df[column] = df[column]

    return winsorized_df.iloc[:, 0] if is_series else winsorized_df

def cluster_data(data: pd.DataFrame, model, run_name: str, mlflow):
    
    results_df = data.copy()
    PCA_components = 2

    mlflow.set_tag("mlflow.runName", run_name)
    mlflow.log_params({"PCA_components": PCA_components})
    mlflow.log_params({"scaler": "RobustScaler"})

    mlflow.autolog()
    results_df['Cluster'] = model.fit_predict(data)

    #Create cache folder if it doesn't exist
    if not path.exists("cache"):
        makedirs("cache")
    
    pca_fig, sum_explained_variance = get_pca(results_df, PCA_components)
    mlflow.log_metrics({"pca_explained_variance": sum_explained_variance})
    pca_fig.savefig("cache/pca_fig.png")
    mlflow.log_artifact("cache/pca_fig.png")

    mlflow.log_metrics({"davies_bouldin_score": davies_bouldin_score(data, results_df['Cluster'])})

    silhouette_fig, silhouette_score = plot_silhouette(data, results_df['Cluster'])
    mlflow.log_metrics({"silhouette_score": silhouette_score})
    silhouette_fig.savefig("cache/silhouette_fig.png")
    mlflow.log_artifact("cache/silhouette_fig.png")

    boxplot_fig = plot_boxplots(results_df, "Cluster")
    boxplot_fig.savefig("cache/boxplot_fig.png")
    mlflow.log_artifact("cache/boxplot_fig.png")
    
    results_df.to_csv("cache/model.csv", index=True)
    mlflow.log_artifact("cache/model.csv")

    return results_df

def plot_boxplots(df, cluster_col: str, max_cols=3):
    """
    Crea un grid de violin plots para cada variable numérica en el DataFrame.

    Parámetros:
    -----------
    df : pandas.DataFrame
        El DataFrame con las variables a graficar.
    figsize : tuple
        Tamaño total de la figura.
    max_cols : int
        Número máximo de columnas en el grid de subplots.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    numeric_cols = np.delete(numeric_cols, numeric_cols.get_loc(cluster_col))
    n_vars = len(numeric_cols)
    
    if n_vars == 0:
        print("No hay variables numéricas para graficar.")
        return

    n_cols = min(n_vars, max_cols)
    n_rows = math.ceil(n_vars / n_cols)
    figsize = (8 * n_cols, 8 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)  # Asegura que axes sea plano

    for i, col in enumerate(numeric_cols):
        sns.boxplot(data=df, y=col, x=cluster_col, ax=axes[i])
        axes[i].set_title(f'Boxplot: {col}')
        axes[i].set_xlabel('Cluster')
    
    # Oculta subplots vacíos
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    return fig