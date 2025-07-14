import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import silhouette_samples, silhouette_score
import seaborn as sns
import numpy as np
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
