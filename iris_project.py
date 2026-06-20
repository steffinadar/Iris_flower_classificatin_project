# ==============================
# IRIS FLOWER CLASSIFICATION
# ==============================

import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

print("="*50)
print("      IRIS FLOWER CLASSIFICATION PROJECT")
print("="*50)

# Load dataset
iris = load_iris()

# Create dataframe
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df['species'] = iris.target
df['species'] = df['species'].replace({
    0:'Setosa',
    1:'Versicolor',
    2:'Virginica'
})

print("\nDataset Loaded Successfully")
print("\nFirst 5 Rows:\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nSpecies Count:")
print(df['species'].value_counts())

# Features and target
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Model...")

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n"+"="*50)
print("MODEL RESULTS")
print("="*50)

print(f"\nAccuracy : {accuracy*100:.2f}%")

print("\nClassification Report:\n")
print(classification_report(
    y_test,
    predictions,
    target_names=iris.target_names
))

# Save model
pickle.dump(
    model,
    open("iris_model.pkl","wb")
)

print("\nModel Saved as iris_model.pkl")

# User prediction
print("\n"+"="*50)
print("FLOWER PREDICTION")
print("="*50)

sl = float(input("Enter Sepal Length: "))
sw = float(input("Enter Sepal Width: "))
pl = float(input("Enter Petal Length: "))
pw = float(input("Enter Petal Width: "))

sample = [[sl, sw, pl, pw]]

result = model.predict(sample)

flower = iris.target_names[result[0]]

print("\nPrediction Result")
print("-"*30)
print("Predicted Flower:", flower.upper())
print("-"*30)

# Confusion matrix
cm = confusion_matrix(
    y_test,
    predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()

plt.title(
    "Confusion Matrix"
)

plt.show()

# Feature importance graph
importance = model.feature_importances_

features = iris.feature_names

plt.figure(figsize=(8,5))

plt.bar(features,importance)

plt.title(
    "Feature Importance"
)

plt.xticks(rotation=20)

plt.show()

print("\nProject Completed Successfully")
print("="*50)
