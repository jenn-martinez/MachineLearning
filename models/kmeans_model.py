import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def ejecutar_kmeans():
    # Cargar dataset (clientes.csv del Kaggle)
    df = pd.read_csv("clientes.csv")

    # === CREAR FEATURES PARA SEGMENTACIÓN ===
    # Agrupar por cliente para crear características de comportamiento
    customer_data = df.groupby('customer_id').agg({
        'quantity': 'sum',  # Total de items comprados
        'price': 'sum',  # Total gastado
        'age': 'first',  # Edad del cliente
        'review_score': 'mean'  # Promedio de reseñas
    }).reset_index()

    # Limpiar valores nulos en review_score
    customer_data['review_score'] = customer_data['review_score'].fillna(customer_data['review_score'].median())

    # Crear columnas con nombres similares a lo que espera el template
    customer_data['Annual Income (k$)'] = customer_data['age'] * 0.8 + 20  # Simular ingreso basado en edad
    customer_data['Spending Score (1-100)'] = (customer_data['price'] / customer_data['price'].max()) * 100

    # Seleccionar features para clustering
    X = customer_data[['Annual Income (k$)', 'Spending Score (1-100)']]

    # Escalado de datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Modelo K-Means (5 clusters)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    customer_data['Cluster'] = kmeans.fit_predict(X_scaled)

    # Centroides en escala original
    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)

    # Resumen por cluster
    resumen = customer_data.groupby('Cluster').agg({
        'Annual Income (k$)': ['mean', 'min', 'max', 'count'],
        'Spending Score (1-100)': ['mean', 'min', 'max']
    }).round(2)

    return customer_data, resumen, centroids_original, X_scaled


def graficar(X, clusters, centroides):
    plt.figure(figsize=(10, 6))

    # Colores para clusters
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    for i in range(len(centroides)):
        cluster_points = X[clusters == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    c=colors[i], label=f'Cluster {i}', alpha=0.6, edgecolors='k')

    # Centroides
    plt.scatter(centroides[:, 0], centroides[:, 1],
                c='black', marker='X', s=200, label='Centroids', edgecolors='white', linewidth=2)

    plt.title("K-Means Clustering - Customer Segmentation", fontsize=14)
    plt.xlabel("Annual Income (k$)", fontsize=12)
    plt.ylabel("Spending Score (1-100)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig("static/graph_kmeans.png", dpi=150, bbox_inches='tight')
    plt.close()


def predecir_cliente(df, income, spending_score):
    """
    Predice a qué cluster pertenece un nuevo cliente
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Crear datos similares a los de entrenamiento
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # Predecir nuevo cliente
    new_customer = np.array([[income, spending_score]])
    new_customer_scaled = scaler.transform(new_customer)
    cluster = kmeans.predict(new_customer_scaled)[0]

    # Centroides
    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    centroid_income = round(centroids_original[cluster][0], 2)
    centroid_score = round(centroids_original[cluster][1], 2)

    # Interpretaciones según el cluster
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