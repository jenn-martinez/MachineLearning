import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_kmeans_data():
    df = pd.read_csv("clientes.csv")

    customer_data = df.groupby('customer_id').agg({
        'quantity': 'sum',
        'price': 'sum',
        'age': 'first',
        'review_score': 'mean'
    }).reset_index()

    customer_data['review_score'] = customer_data['review_score'].fillna(customer_data['review_score'].median())

    customer_data['Annual Income (k$)'] = customer_data['age'] * 0.8 + 20
    customer_data['Spending Score (1-100)'] = (customer_data['price'] / customer_data['price'].max()) * 100

    X = customer_data[['Annual Income (k$)', 'Spending Score (1-100)']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    customer_data['Cluster'] = kmeans.fit_predict(X_scaled)

    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)

    resumen = customer_data.groupby('Cluster').agg({
        'Annual Income (k$)': ['mean', 'min', 'max', 'count'],
        'Spending Score (1-100)': ['mean', 'min', 'max']
    }).round(2)

    # Métricas completas
    metrics = {
        "inertia": round(kmeans.inertia_, 2),
        "silhouette": round(silhouette_score(X_scaled, customer_data['Cluster']), 4),
        "davies_bouldin": round(davies_bouldin_score(X_scaled, customer_data['Cluster']), 4),
        "calinski": round(calinski_harabasz_score(X_scaled, customer_data['Cluster']), 2)
    }

    # Generar gráfico
    graficar(X_scaled, customer_data['Cluster'].values, centroids_scaled)

    # Conteo por cluster
    cluster_counts = customer_data['Cluster'].value_counts().sort_index().to_dict()

    return {
        "customer_data": customer_data,
        "resumen": resumen.to_html(classes='table table-sm table-striped'),
        "tabla": customer_data[['customer_id', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']].head(20).to_html(classes='table table-sm table-striped', index=False),
        "centroids": centroids_original.tolist(),
        "X_scaled": X_scaled.tolist(),
        "metrics": metrics,
        "n_clusters": 5,
        "total_records": len(customer_data),
        "cluster_counts": cluster_counts
    }


def graficar(X, clusters, centroides):
    plt.figure(figsize=(10, 6))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    for i in range(len(centroides)):
        cluster_points = X[clusters == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    c=colors[i], label=f'Cluster {i}', alpha=0.6, edgecolors='k')

    plt.scatter(centroides[:, 0], centroides[:, 1],
                c='black', marker='X', s=200, label='Centroids', edgecolors='white', linewidth=2)

    plt.title("K-Means Clustering - Customer Segmentation", fontsize=14)
    plt.xlabel("Annual Income (k$)", fontsize=12)
    plt.ylabel("Spending Score (1-100)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig("static/graph_kmeans.png", dpi=150, bbox_inches='tight')
    plt.close()


def predict_customer(df, income, spending_score):
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    new_customer = np.array([[income, spending_score]])
    new_customer_scaled = scaler.transform(new_customer)
    cluster = kmeans.predict(new_customer_scaled)[0]

    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    centroid_income = round(centroids_original[cluster][0], 2)
    centroid_score = round(centroids_original[cluster][1], 2)

    interpretations = {
        0: "Standard Customer - Moderate income and spending",
        1: "Premium Customer - High income, high spending",
        2: "Budget Customer - Low income, low spending",
        3: "Potential Customer - High income, low spending",
        4: "Occasional Customer - Low income, moderate spending"
    }

    strategies = {
        0: "Standard marketing with occasional promotions",
        1: "Loyalty programs and premium benefits",
        2: "Discount coupons and value offers",
        3: "Targeted campaigns to increase spending",
        4: "Engagement campaigns to increase frequency"
    }

    return {
        'cluster': int(cluster),
        'label': interpretations.get(cluster, f"Cluster {cluster}"),
        'strategy': strategies.get(cluster, "Standard marketing approach"),
        'centroid_income': centroid_income,
        'centroid_score': centroid_score
    }