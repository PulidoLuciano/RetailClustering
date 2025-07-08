import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

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

