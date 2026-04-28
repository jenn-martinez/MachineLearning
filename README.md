Machine Learning Use Cases
Description
This project is a web application built with Flask that demonstrates real-world Machine Learning use cases through an interactive interface. Users can explore both supervised and unsupervised learning techniques, visualize model behavior, and make predictions in real time.

The application covers four ML algorithms: Linear Regression, Logistic Regression, Perceptron (Binary Classification), and K-Means Clustering, each backed by a real dataset and an interactive UI module.

Objectives
Understand the fundamentals of supervised and unsupervised Machine Learning.

Implement and compare different ML models (regression, classification, clustering).

Integrate trained models into a Flask web application.

Visualize data, decision boundaries, and prediction results interactively.

Apply real datasets to solve practical problems.

Machine Learning Concepts Covered
Supervised Learning

Linear Regression

Logistic Regression

Perceptron (Binary Classification)

Unsupervised Learning

K-Means Clustering

Data Preprocessing & Feature Engineering

Model Training, Evaluation, and Prediction

Data Visualization (Matplotlib, ROC Curve, Confusion Matrix)

Use Cases Included
Use Case	Description
🏦 Bank Fraud Detection	Conceptual overview of fraud detection pipelines
🏢 BBVA Predictive Model	Predictive model for debt resolution
👤 Facial Recognition	Overview of facial recognition with ML
📉 Customer Churn Prediction	Predicting customer churn with classification
ML Modules
📈 Linear Regression — Olympic Medals Dataset
Predicts the total number of medals based on a selected numeric variable (e.g., Gold, Silver, Bronze medals per country).

Dataset: medals.csv

Source: Kaggle

Features: Country performance metrics per Olympic event

Target: Total medal count

Output: Regression line graph + real-time prediction from user input

🎮 Logistic Regression — FIFA 24 Players Dataset
Classifies FIFA 24 players based on their in-game stats to predict a categorical outcome.

Dataset: fifa24.csv

Source: Kaggle

Features: pace, shooting, passing, dribbling

Target: Player classification (binary)

Output: Classification result with input form

🔬 Perceptron (Classification) — Breast Cancer Wisconsin Dataset
Binary classification model that predicts whether a tumor is malignant or benign based on cell nucleus measurements.

Dataset: breastCancerWisconsin.csv

Source: UCI Machine Learning Repository / Kaggle

Features: 30 numeric features (radius, texture, perimeter, area, smoothness, etc.)

Target: diagnosis — M (Malignant) or B (Benign)

Output: Confusion Matrix, ROC Curve, error-per-epoch graph, and live prediction

🛍️ K-Means Clustering — E-Commerce Customers Dataset
Groups customers into segments based on their purchasing behavior using unsupervised learning.

Dataset: clientes.csv

Source: Synthetic e-commerce dataset

Columns:

Column	Description
customer_id	Unique customer identifier
order_date	Date of purchase
product_id	Unique product identifier
category_id	Product category ID
category_name	Category name (Electronics, Fashion, Sports & Outdoors, etc.)
product_name	Name of the purchased product
quantity	Number of units purchased
price	Unit price
payment_method	Payment method (Credit Card, Bank Transfer, Cash on Delivery)
city	Customer's city
review_score	Customer review score (1–5, may contain nulls)
gender	Customer gender (M/F, may contain nulls)
age	Customer age
Objective: Customer segmentation — identify groups by purchase patterns, spending habits, and demographics.

Output: Cluster visualization, manual exercise, and concept overview

Technologies Used
Technology	Purpose
Python 3	Core language
Flask	Web framework
Pandas	Data manipulation
NumPy	Numerical computation
Scikit-learn	Model training and evaluation
Matplotlib	Data visualization
HTML & CSS	Frontend templates
Project Structure
text
MachineLearning/
├── app.py                        # Main Flask application
├── clientes.csv                  # K-Means dataset (customer e-commerce data)
├── medals.csv                    # Linear Regression dataset
├── fifa24.csv                    # Logistic Regression dataset
├── breastCancerWisconsin.csv     # Perceptron dataset
├── models/
│   ├── linearRegression.py
│   ├── logisticRegression.py
│   └── perceptron_model.py
└── templates/
    ├── mainMenu.html
    ├── caseUse/
    ├── linearRegression/
    ├── logisticRegression/
    ├── classificationPerceptron/
    └── kMeans/
Installation and Execution
Clone the repository:

bash
git clone https://github.com/jenn-martinez/MachineLearning.git
cd MachineLearning
Install dependencies:

bash
pip install flask pandas numpy scikit-learn matplotlib
Run the application:

bash
python app.py
Open in browser:

text
http://127.0.0.1:5000/
Future Improvements
Add more advanced models (Random Forest, SVM, Neural Networks)

Expand the K-Means module with elbow method and silhouette analysis

Improve the UI/UX design of all modules

Add real-time data upload and processing

Deploy the application to a cloud platform (Heroku, Render, etc.)

Authors
Santiago Bustos Coca

Jennyfer Martinez Vargas

Sergio Juyo

Juan Medina

Repository
https://github.com/jenn-martinez/MachineLearning