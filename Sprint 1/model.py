"""
Model Problem
Sprint 1 
Efe Comu
"""

import pandas as pd
import numpy as np

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Open the Titanic dataset
df = pd.read_csv('Titanic.csv')

# Print the first few lines of the dataframe
#
# Check the names and types of each column
print(df.head())

# scikit-learn can't work with arbitrary strings, like "male" and "female"
# Create a new field that turns the "Sex" string into a binary number
df['Sex_numeric'] = np.where(df['Sex'] == 'male', 1, 0)

# Pull out the Pclass, Age, and Sex_numeric columns into their own dataframe
X = df[['Pclass', 'Age', 'Sex_numeric']]

# Extract the target variable
y = df.pop('Survived').values
print(y)

# Create the Decision TreeClassifier object
# max_depth=2 restricts the size of the tree to only two splits
clf = tree.DecisionTreeClassifier(max_depth=20)

# Split the complete data set into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)

# Fit the decision tree
# Our goal is to predict Survival based on Pclass, Age, and Sex
clf.fit(X_train, y_train)

tree.export_graphviz(clf,
                     out_file='tree.dot',
                     feature_names=['Pclass', 'Age', 'Sex_numeric'],
                     class_names=['Did not survive', 'Survived'],
                     filled=True,
                     rounded=True,
                     special_characters=True,
                     proportion=True)

# Test the fitted model on the testing subset
y_pred = clf.predict(X_test)

# Output the accuracy of the model
# Calculated by comparing the predicted classifications of the test set to the
# correct classifications
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
