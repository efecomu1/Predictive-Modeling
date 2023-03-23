"""
Analyzing the Disputed Federalist Papers
"""

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import csv
df = pd.read_csv('federalist_papers_data.csv')

a = df[df['class'] == 1].count()
print(a)

# Separate the entire dataframe into two subsets
# One for training, containing the papers of known authorship
# The other for testing, which is the papers of unknown authorship
train_df = df[df['class'] != 3]
test_df = df[df['class'] == 3]

print(test_df)
print(train_df)

# Extract the target variable
y = train_df.pop('class').values
print(y)
print(train_df)

# Create the Decision TreeClassifier object
# Fit the decision tree
clf = tree.DecisionTreeClassifier(max_depth=50)
clf.fit(train_df, y)

# Visualize the decision tree,
tree.export_graphviz(clf,
                     out_file='tree.dot',
                     feature_names=list(train_df.columns),
                     class_names=['Hamilton', 'Madison'],
                     filled=True,
                     rounded=True,
                     special_characters=True,
                     proportion=True)

# Use the fitted model to predict the class of the unknown papers
y = test_df.pop('class').values
predicted_labels = clf.predict(test_df)
print(predicted_labels)

#######
# Writeup
# In this problem, the data regarding the collection of short essays written, known as "The Federalist Papers", by Alexander Hamilton or James Madison was analyzed to identify the authors of the disputed papers in our dataset by using a decision tree.
# Our dataset contains 118 data points, 56 of them belonging to Hamilton's papers, 50 of them belonging to Madison's papers, and 12 of them being unknown disputed papers. The data points in the dataset also contain frequency of occurence of the 70 function words in the text, scaled to units of estimated occurences per 1000 words to eliminate bias caused by the length of the essays. To train the model, a dataframe called "train_df" was initialized, having all the data points that have a stated author (class 1 (Hamilton) & class 2 (Madison)). The values of class (1 or 2) are extracted from this dataframe in order to use it as a target variable in our decision tree, and the "train_df" was used to build the decision tree in order to identify the words that distinguish these two authors the most.
# As the decision tree that is built based on our training data suggests, the word that gives our model the biggest information gain is the frequency of the word "upon". This can be seen in the top node of our decision tree that contains 100% of our training data. The nodes of the first split proves this point, as it shows that more than 98% of essays that have a frequency of more than 1.34 for the word "upon" belongs to Hamilton, and more than 94% of essays that have a frequency of less than 1.34 for the word "upon" belongs to Madison. The other words that were chosen as significant by our model were "it", "is", "our", and "info", leading to our decision tree growing until it reaches the gini coefficient of 0 for every node.
# According to Bosch, Smith, and Fung, all of the 12 disputed papers were works of Madison, but there are still opinions of other scholars that believe that some of these works are Hamilton's pieces or collaborative efforts. To solve this uncertainty, we used our decision tree classifier model to test our "test_df" data consisting of all 12 disputed papers. As a results, our model attributed 10/12 papers to Madison, and 2/12 papers to Hamilton. Results: [2 2 1 2 2 2 2 2 1 2 2 2]
# In conclusion, the purpose of our study was to train a decision tree model based on the certain existing data to predict the authors of the 12 disputed papers in our dataset. To do this, a decision tree with a max_depth of 3 is visualized, and "upon" is identified as the word frequency that distinguishes these two authors the most. Finally, the results of our study were 10 papers belonging to Madison, and 2 papers belonging to Hamilton, unlike the theories of Bosch, Smith, and Fung.
