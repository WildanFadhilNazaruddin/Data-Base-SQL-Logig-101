import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load and preprocess the dataset
data = pd.read_csv('dataset.csv')
data.fillna(0, inplace=True)

# Define features and target
X = data.drop('target', axis=1)
y = data['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the machine learning model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the dataset
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'trained_model.pkl')
