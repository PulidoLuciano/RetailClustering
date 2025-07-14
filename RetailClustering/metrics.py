from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score


class ClusteringMetrics:

    def __init__(self, scaled_df: pd.DataFrame, verbose=False) -> None:
        self._df = scaled_df
        self._verbose = verbose

    # === Funciones de Métricas ===

    def __dunn_index(self, X: np.ndarray, labels: np.ndarray) -> np.float64:
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        if n_clusters < 2:
            return np.float64(np.nan)  # Dunn no es válido si hay solo un cluster

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
            return np.float64(np.nan)

        dunn = np.min(inter_cluster_dists) / np.max(intra_cluster_diameters)
        return dunn

    def __apn(self, labels_full: np.ndarray, labels_minus: np.ndarray) -> np.float64:
        """
            APN (Average Proportion of Non-overlap)
            Mide cuántas observaciones cambian de cluster al quitar una columna.
            
            Parameters:
                labels_full (np.ndarray): Etiquetas del clustering con todas las columnas.
                labels_minus (np.ndarray): Etiquetas del clustering con una columna eliminada.
            
            Returns:
                np.float64: Proporción promedio de observaciones que cambiaron de cluster.
        """
        if labels_full.shape != labels_minus.shape:
            raise ValueError("label arrays must have the same shape")
            
        n = len(labels_full)
        overlap = np.sum(labels_full == labels_minus)
        return np.float64(1 - overlap / n)

    def __intra_cluster_distances(self, X: np.ndarray, labels: np.ndarray) -> np.float64:
        """
            AD (Average Distance)
            Mide la distancia promedio entre puntos de un mismo cluster.

            Parameters:
                X (np.ndarray): Datos originales (n_samples, n_features).
                labels (np.ndarray): Etiquetas de clustering.

            Returns:
                np.float64: Promedio de distancias intra-cluster.
        """
        unique_labels = np.unique(labels)
        intra_dists = []

        for label in unique_labels:
            cluster_points = X[labels == label]
            if len(cluster_points) > 1:
                dists = pairwise_distances(cluster_points)
                # Solo considerar la mitad superior de la matriz (sin diagonales)
                upper_tri_indices = np.triu_indices_from(dists, k=1)
                intra_dists.append(np.mean(dists[upper_tri_indices]))

        return np.float64(np.mean(intra_dists)) if intra_dists else np.float64(0.0)

    def __average_distance_between_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.float64: 
        """
            ADM (Average Distance between Means)
            Mide la distancia promedio entre los centroides de los clusters.

            Parameters:
                X (np.ndarray): Datos originales (n_samples, n_features).
                labels (np.ndarray): Etiquetas de clustering.

            Returns:
                np.float64: Distancia promedio entre centroides.
        """
        unique_labels = np.unique(labels)
        centroids = np.array([X[labels == label].mean(axis=0) for label in unique_labels])

        if len(centroids) < 2:
            return np.float64(0.0)

        dists = pairwise_distances(centroids)
        upper_tri_indices = np.triu_indices_from(dists, k=1)

        return np.float64(np.mean(dists[upper_tri_indices]))

    def __fom(self, X, labels, metric='euclidean', use_silhouette=False):
        """
        Computes a custom figure of merit (FOM) for clustering quality.

        Parameters:
            X (array-like): The dataset (n_samples x n_features)
            labels (array-like): Cluster labels for each sample
            metric (str): Distance metric for pairwise calculations
            use_silhouette (bool): If True, returns the silhouette score as FOM.
                                If False, returns a custom compactness/separation metric.

        Returns:
            float: The computed FOM (higher is better)
        """

        if use_silhouette:
            # Higher silhouette score means better-defined clusters
            return silhouette_score(X, labels, metric=metric)

        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        if n_clusters < 2:
            return -1  # FOM not meaningful with <2 clusters

        # Compute centroids
        centroids = np.array([X[labels == label].mean(axis=0) for label in unique_labels])

        # Compute intra-cluster distances (compactness)
        intra_dists = []
        for label in unique_labels:
            cluster_points = X[labels == label]
            centroid = cluster_points.mean(axis=0)
            dists = pairwise_distances(cluster_points, [centroid], metric=metric)
            intra_dists.append(np.mean(dists))

        avg_intra_dist = np.mean(intra_dists)

        # Compute inter-cluster distances (separation)
        inter_dists = pairwise_distances(centroids, metric=metric)
        mask = ~np.eye(n_clusters, dtype=bool)
        avg_inter_dist = np.mean(inter_dists[mask])

        # Figure of merit = separation / compactness
        fom = avg_inter_dist / avg_intra_dist if avg_intra_dist != 0 else 0

        return fom

    # === Generacion de metricas por algoritmo de clustering 
    def get_metrics(self, model):
        X_full = self._df.values
        features = self._df.columns.tolist()

        labels_full = model.fit_predict(X_full)

        apn_list = []
        ad_list = []
        adm_list = []
        fom_list = []

        for i, col in enumerate(features):
            X_minus = self._df.drop(columns=[col]).values
            labels_minus = model.fit_predict(X_minus)

            apn_val = self.__apn(labels_full, labels_minus)
            ad_val = abs(self.__intra_cluster_distances(X_full, labels_full) - 
                        self.__intra_cluster_distances(X_minus, labels_minus))
            adm_val = abs(self.__average_distance_between_centroids(X_full, labels_full) -
                        self.__average_distance_between_centroids(X_minus, labels_minus))

            column_values = self._df[col].values
            cluster_var = []
            for label in np.unique(labels_minus):
                if label == -1:
                    continue
                cluster_points = column_values[labels_minus == label]
                if len(cluster_points) > 1:
                    cluster_var.append(np.var(cluster_points))
            fom_val = np.mean(cluster_var) if cluster_var else np.nan

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
            print(f"\t FOM: \t {np.nanmean(fom_list):.4f}")
            print(f"\t Dunn Index: {dunn:.4f}")

        return {
            "apn": round(np.mean(apn_list), 4),
            "ad": round(np.mean(ad_list), 4),
            "adm": round(np.mean(adm_list), 4),
            "fom": round(np.nanmean(fom_list), 4) ,
            "dunn": round(dunn, 4)
        }

    def KMeans(self, n_clusters=4, random_state=42) -> dict:
        return self.get_metrics(KMeans(n_clusters=n_clusters, random_state=random_state))

    def DBSCAN(self, eps=0.5, min_samples=5) -> dict:
        return self.get_metrics(DBSCAN(eps=eps, min_samples=min_samples))

    def AgglomerativeClustering(self, n_clusters=4) -> dict:
        return self.get_metrics(AgglomerativeClustering(n_clusters=n_clusters))

# Testing with mock dataset.
# blobs_data = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=42)
# X = blobs_data[0]
# X_scaled = StandardScaler().fit_transform(X)
# scaled_df = pd.DataFrame(X_scaled)

# cm = ClusteringMetrics(scaled_df, verbose=True)
# result_kmeans = cm.kmeans()
# result_dbscan = cm.dbscan()
# result_agglomerative = cm.agglomerative()

