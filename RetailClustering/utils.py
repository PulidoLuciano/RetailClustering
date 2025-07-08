import pandas as pd

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