from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

class ClusteringMetrics:

    def __init__(self, scaled_df: pd.DataFrame, verbose=False) -> None:
        self._df = scaled_df
        self._verbose = verbose

    # === Funciones de Métricas ===

    def __dunn_index(self, X, labels):
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        if n_clusters < 2:
            return np.nan  # Dunn no es válido si hay solo un cluster

        inter_cluster_dists = []
        intra_cluster_diameters = []

        for i, label_i in enumerate(unique_labels):
            cluster_i = X[labels == label_i]

            # Intra-cluster: diámetro del cluster (máxima distancia entre pares)
            if len(cluster_i) > 1:
                intra_dists = pairwise_distances(cluster_i)
                diameter = np.max(intra_dists)
            else:
                diameter = 0  # cluster con un solo punto
            intra_cluster_diameters.append(diameter)

            for j, label_j in enumerate(unique_labels):
                if j <= i:
                    continue
                cluster_j = X[labels == label_j]

                # Inter-cluster: mínima distancia entre puntos de dos clusters
                inter_dists = pairwise_distances(cluster_i, cluster_j)
                min_dist = np.min(inter_dists)
                inter_cluster_dists.append(min_dist)

        if len(inter_cluster_dists) == 0 or max(intra_cluster_diameters) == 0:
            return np.nan

        dunn = np.min(inter_cluster_dists) / np.max(intra_cluster_diameters)
        return dunn

    def __apn(self, labels_full, labels_minus):
        '''  APN (Average Proportion of Non-overlap)
        - Mide cuantas observaciones cambian de cluster al quitar una columna

        labels_full: Dataframe con todas las columnas.
        lavels_minus: Dataframe con todas las columnas menos una.

        '''
        n = len(labels_full)
        overlap = sum(labels_full[i] == labels_minus[i] for i in range(n))
        return 1 - overlap / n

    def __intra_cluster_distances(self, X, labels):
        ''' AD (Average Distance)
        - Mide la distancia promedio entre puntos de un mismo cluster. 
        Se calcula para X completo y para cada X_minus_i. Luego se promedia la diferencia.

        X: Dataframe con todas las columnas.
        labels: Dataframe con todas las columnas menos una.

        '''
        unique_labels = np.unique(labels)
        distances = []
        for label in unique_labels:
            cluster_points = X[labels == label]
            if len(cluster_points) > 1:
                d = pairwise_distances(cluster_points)
                distances.append(np.mean(d))
        return np.mean(distances)

    def __average_distance_between_centroids(self, X, labels):
        ''' ADM (Average Distance between Means)
        Qué mide: la distancia entre los centroides (medias) de los clusters con y 
        sin una columna.

        '''
        unique_labels = np.unique(labels)
        centroids = [X[labels == label].mean(axis=0) for label in unique_labels]
        d = pairwise_distances(centroids)
        return np.mean(d)

    def __fom(self, X_column, labels):
        ''' FOM (Figure of Merit)
        Qué mide: la varianza de la columna eliminada, dentro de cada 
        cluster calculado sin esa columna.

        '''
        unique_labels = np.unique(labels)
        fom_vals = []
        for label in unique_labels:
            vals = X_column[labels == label]
            if len(vals) > 1:
                fom_vals.append(np.var(vals))
        return np.mean(fom_vals)

    # === Generacion de metricas por algoritmo de clustering 

    def kmeans(self, n_clusters=4, random_state=42) -> dict:
        X_full = self._df.values
        features = self._df.columns.tolist()

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels_full = kmeans.fit_predict(X_full)

        apn_list = []
        ad_list = []
        adm_list = []
        fom_list = []


        for i, col in enumerate(features):
            X_minus = scaled_df.drop(columns=[col]).values
            labels_minus = KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(X_minus)

            apn_val = self.__apn(labels_full, labels_minus)
            ad_val = abs(self.__intra_cluster_distances(X_full, labels_full) - 
                        self.__intra_cluster_distances(X_minus, labels_minus))
            adm_val = abs(self.__average_distance_between_centroids(X_full, labels_full) -
                        self.__average_distance_between_centroids(X_minus, labels_minus))
            fom_val = self.__fom(scaled_df[col].values, labels_minus)

            apn_list.append(apn_val)
            ad_list.append(ad_val)
            adm_list.append(adm_val)
            fom_list.append(fom_val)

        dunn = self.__dunn_index(X_full, labels_full)

        # aqui hacemos resultados promedio
        if self._verbose:
            print(f"KMeans: ")
            print(f"\t APN: \t {np.mean(apn_list):.4f}")
            print(f"\t AD: \t {np.mean(ad_list):.4f}")
            print(f"\t ADM: \t {np.mean(adm_list):.4f}")
            print(f"\t FOM: \t {np.mean(fom_list):.4f}")
            print(f"\t Dunn Index: {dunn:.4f}")

        return {
            "apn": round(np.mean(apn_list), 4),
            "ad": round(np.mean(ad_list), 4),
            "adm": round(np.mean(adm_list), 4),
            "fom": round(np.mean(fom_list), 4) ,
            "dunn": round(dunn, 4)
        }

    def dbscan(self, eps=0.5, min_samples=5) -> dict:

        X_full = self._df.values
        features = self._df.columns.tolist()

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels_full = dbscan.fit_predict(X_full)

        apn_list = []
        ad_list = []
        adm_list = []
        fom_list = []

        for i, col in enumerate(features):
            X_minus = scaled_df.drop(columns=[col]).values
            labels_minus = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_minus)

            apn_val = self.__apn(labels_full, labels_minus)
            ad_val = abs(self.__intra_cluster_distances(X_full, labels_full) - 
                        self.__intra_cluster_distances(X_minus, labels_minus))
            adm_val = abs(self.__average_distance_between_centroids(X_full, labels_full) -
                        self.__average_distance_between_centroids(X_minus, labels_minus))
            fom_val = self.__fom(scaled_df[col].values, labels_minus)

            apn_list.append(apn_val)
            ad_list.append(ad_val)
            adm_list.append(adm_val)
            fom_list.append(fom_val)

        # Par DBSCAN, el dunn index necesita que hagamos limpieza de ruido
        mask = labels_full != -1
        X_clean = X_full[mask]
        labels_clean = labels_full[mask]
        dunn = self.__dunn_index(X_clean, labels_clean)

        # aqui hacemos resultados promedio (ignorando NaN si hubo ruido total)
        if self._verbose:
            print(f"DBSCAN: ")
            print(f"\t APN: \t {np.nanmean(apn_list):.4f}")
            print(f"\t AD: \t {np.nanmean(ad_list):.4f}")
            print(f"\t ADM: \t {np.nanmean(adm_list):.4f}")
            print(f"\t FOM: \t {np.nanmean(fom_list):.4f}")
            print(f"\t Dunn Index: {dunn:.4f}")

        return {
            "apn": round(np.nanmean(apn_list), 4),
            "ad": round(np.nanmean(ad_list), 4),
            "adm": round(np.nanmean(adm_list), 4),
            "fom": round(np.nanmean(fom_list), 4),
            "dunn": round(dunn, 4),
        }

    def agglomerative(self, n_clusters=4) -> dict:
        X_full = self._df.values
        features = self._df.columns.tolist()

        agglo = AgglomerativeClustering(n_clusters=n_clusters)
        labels_full = agglo.fit_predict(X_full)

        apn_list = []
        ad_list = []
        adm_list = []
        fom_list = []

        for i, col in enumerate(features):
            X_minus = scaled_df.drop(columns=[col]).values
            labels_minus = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X_minus)

            apn_val = self.__apn(labels_full, labels_minus)
            ad_val = abs(self.__intra_cluster_distances(X_full, labels_full) - 
                        self.__intra_cluster_distances(X_minus, labels_minus))
            adm_val = abs(self.__average_distance_between_centroids(X_full, labels_full) -
                        self.__average_distance_between_centroids(X_minus, labels_minus))
            fom_val = self.__fom(scaled_df[col].values, labels_minus)

            apn_list.append(apn_val)
            ad_list.append(ad_val)
            adm_list.append(adm_val)
            fom_list.append(fom_val)

        dunn = self.__dunn_index(X_full, labels_full)

        # resultados promedio
        if self._verbose:
            print(f"Agglomerative: ")
            print(f"\t APN: \t {np.mean(apn_list):.4f}")
            print(f"\t AD: \t {np.mean(ad_list):.4f}")
            print(f"\t ADM: \t {np.mean(adm_list):.4f}")
            print(f"\t FOM: \t {np.mean(fom_list):.4f}")
            print(f"\t Dunn Index: {dunn:.4f}")

        return {
            "apn": round(np.nanmean(apn_list), 4),
            "ad": round(np.nanmean(ad_list), 4),
            "adm": round(np.nanmean(adm_list), 4),
            "fom": round(np.nanmean(fom_list), 4),
            "dunn": round(dunn, 4),
        }

# Testing with mock dataset.
blobs_data = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=42)
X = blobs_data[0]
X_scaled = StandardScaler().fit_transform(X)
scaled_df = pd.DataFrame(X_scaled)

cm = ClusteringMetrics(scaled_df, verbose=True)
result_kmeans = cm.kmeans()
result_dbscan = cm.dbscan()
result_agglomerative = cm.agglomerative()

