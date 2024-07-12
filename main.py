# This is a Python script for Machine Learning Decision Trees to determine if a transaction is fraud.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree
import graphviz

# Create the path to the dataset and set dataframe to the data
path_to_file = "../CSC 419 HW 2 Transaction Data.csv"
df = pd.read_csv(path_to_file)
# print(df.head())
# print(df.info)

# Split the dataset into Train-Test
X = df.drop('fraud', axis=1)
y = df[['fraud']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create Decision Tree Classifier and fit to model - Tree 1
clf_model = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=3, min_samples_leaf=5)
clf_model.fit(X_train, y_train)

# Make Predictions
y_predict = clf_model.predict(X_test)

# Test accuracy
accuracy = accuracy_score(y_test, y_predict)
print("T1 Accuracy:", accuracy)

# Plot Decision Tree
target = ['no', 'yes']
feature_names = list(X.columns)
dot_data = tree.export_graphviz(clf_model,
                                out_file=None,
                                feature_names=feature_names,
                                class_names=target,
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("C:/Users/Cudlino/Documents/CUW/Summer 2024 Courses/Machine Learning/Unit 2/DecisionTree1.dot")
r = export_text(clf_model, feature_names=feature_names)
print(r)

# Create Decision Tree Classifier and fit to model - Tree 2
clf_model2 = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=2, min_samples_leaf=5)
clf_model2.fit(X_train, y_train)

# Make Predictions
y_predict2 = clf_model2.predict(X_test)

# Test accuracy
accuracy = accuracy_score(y_test, y_predict2)
print("T2 Accuracy:", accuracy)

# Plot Decision Tree
target = ['no', 'yes']
feature_names = list(X.columns)
dot_data2 = tree.export_graphviz(clf_model2,
                                out_file=None,
                                feature_names=feature_names,
                                class_names=target,
                                filled=True, rounded=True,
                                special_characters=True)
graph2 = graphviz.Source(dot_data2)

graph2.render("C:/Users/Cudlino/Documents/CUW/Summer 2024 Courses/Machine Learning/Unit 2/DecisionTree2.dot")

r2 = export_text(clf_model, feature_names=feature_names)
print(r2)