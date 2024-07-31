#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score

# Load the data
data = pd.read_excel('Mental disorder symptoms.xlsx')

# Clean and preprocess the data
mental_data = data.rename(columns={'ag+1:629e': 'age', 'Disorder': 'disorder'})

# Check for missing values and duplicates
print(mental_data.isna().sum())
print(mental_data.duplicated().sum())

mental_data.drop_duplicates(inplace=True)

# Plotting
plt.figure(figsize=(5, 5))
sns.histplot(mental_data['age'], bins=10, kde=True)
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(5, 5))
plt.scatter(mental_data['age'], mental_data['disorder'])
plt.title('Age vs Disorder')
plt.xlabel('Age')
plt.ylabel('Disorder')
plt.show()

plt.figure(figsize=(15, 10))
sns.countplot(x='disorder', data=mental_data)
plt.title('Disorder Count')
plt.xticks(rotation=45)
plt.show()

# Encode labels
label_encoder = LabelEncoder()
mental_data['disorder'] = label_encoder.fit_transform(mental_data['disorder'])

# Prepare the data for modeling
X = mental_data.drop(['disorder', 'age'], axis=1)
y = mental_data['disorder']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Model training and evaluation
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42)
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f'---{model_name}---')
    print(f'Accuracy: {accuracy_score(y_test, predictions)}')
    print(f'F1 Score: {f1_score(y_test, predictions, average="weighted")}')
    print(f'Recall: {recall_score(y_test, predictions, average="weighted")}')
    print(classification_report(y_test, predictions))

# Model performance comparison
performance_data = {
    'Model': list(models.keys()),
    'Accuracy': [accuracy_score(y_test, model.predict(X_test)) for model in models.values()],
    'F1 Score': [f1_score(y_test, model.predict(X_test), average="weighted") for model in models.values()],
    'Recall': [recall_score(y_test, model.predict(X_test), average="weighted") for model in models.values()]
}

performance_df = pd.DataFrame(performance_data)
performance_df.set_index('Model').plot(kind='bar', figsize=(10, 6))
plt.title('Model Performance')
plt.ylabel('Score')
plt.show()

# Hyperparameter tuning for RandomForest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search_rf = GridSearchCV(estimator=models['RandomForest'], param_grid=param_grid, cv=3, scoring='f1_weighted')
grid_search_rf.fit(X_train, y_train)

best_rf_model = grid_search_rf.best_estimator_
best_rf_predictions = best_rf_model.predict(X_test)

print(f'---Tuned RandomForest---')
print(f'Accuracy: {accuracy_score(y_test, best_rf_predictions)}')
print(f'F1 Score: {f1_score(y_test, best_rf_predictions, average="weighted")}')
print(f'Recall: {recall_score(y_test, best_rf_predictions, average="weighted")}')
print(classification_report(y_test, best_rf_predictions))

# Tuned model performance
tuned_performance_data = {
    'Model': ['Tuned RandomForest'],
    'Accuracy': [accuracy_score(y_test, best_rf_predictions)],
    'F1 Score': [f1_score(y_test, best_rf_predictions, average="weighted")],
    'Recall': [recall_score(y_test, best_rf_predictions, average="weighted")]
}

tuned_performance_df = pd.DataFrame(tuned_performance_data)
tuned_performance_df.set_index('Model').plot(kind='bar', figsize=(10, 6))
plt.title('Tuned Model Performance')
plt.ylabel('Score')
plt.show()

# Save the model
pickle.dump(best_rf_model, open('best_rf_model.pkl', 'wb'))
loaded_model = pickle.load(open('best_rf_model.pkl', 'rb'))