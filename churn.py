"""
Churn Challenge Project
Sprint 3 
Efe Comu
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import the .csv file into a dataframe
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn 2.csv')

df['Churned_bool'] = np.where(df['Churn'] == "Yes", 1, 0)
print(df.head())

churned = df[df['Churned_bool'] == 1]
print(churned)

# First feature: Senior Citizen fraction
a = churned[df['SeniorCitizen'] == 0].count()
# print(a)  # 1393
b = churned[df['SeniorCitizen'] == 1].count()
# print(b)  # 476

non_senior_citizens = df[df['SeniorCitizen'] == 0]
# print(non_senior_citizens.count())
# 5901

senior_citizens = df[df['SeniorCitizen'] == 1]
# print(senior_citizens.count())
# 1142

# bar graph
senior_frac = 476 / 1142
nonsenior_frac = 1393 / 5901

plt.figure()
x_axis = ['Senior citizens', 'Non-senior citizens']
y_axis = [senior_frac, nonsenior_frac]
plt.bar(x_axis, y_axis, color=[
  'red',
  'green',
])
plt.title(
  "Fraction of senior citizens that churned vs. non-senior citizens that churned"
)
plt.ylabel("Fraction of churn")
plt.ylim([0, 1])
plt.savefig('senior_citizens.png', bbox_inches='tight')
plt.close()

# Second feature: Gender
df['gender_bool'] = np.where(df['gender'] == "Male", 1, 0)
# print(df.head())
female = churned[df['gender_bool'] == 0].count()
# print(female)  # 939
male = churned[df['gender_bool'] == 1].count()
# print(male)  # 930

females_total = df[df['gender_bool'] == 0]
# print(females_total.count())
# 3488

males_total = df[df['gender_bool'] == 1]
# print(males_total.count())
# 3555

# bar graph
females_frac = 939 / 3488
males_frac = 930 / 3555

plt.figure()
x_axis = ['Female', 'Male']
y_axis = [females_frac, males_frac]
plt.bar(x_axis, y_axis, color=[
  'red',
  'blue',
])
plt.title("Fraction of females that churned vs. males that churned")
plt.ylabel("Fraction of churn")
plt.ylim([0, 1])
plt.savefig('gender.png', bbox_inches='tight')
plt.close()

### Brief explanation
# As seen the 2 bar charts as result of the explanatory analysis, it can be said that whether a customer is a senior citizen or not has an effect on the churning rate, whereas gender has no significant effect on the churn outcome.

# Remove the customer ID from the dataset before starting to construct the model
df = df.drop('customerID', axis=1)
print(df.head())

df['Partner_bool'] = np.where(df['Partner'] == "Yes", 1, 0)
df['Dependents_bool'] = np.where(df['Dependents'] == "Yes", 1, 0)
df['PhoneService_bool'] = np.where(df['PhoneService'] == "Yes", 1, 0)

# Pull out the Churn column as a target variable
X = df[[
  'gender_bool', 'SeniorCitizen', 'Partner_bool', 'Dependents_bool',
  'PhoneService_bool', 'tenure', 'MonthlyCharges'
]]
y = df.pop('Churn').values

# Hold out 20-30% of the data as a testing set, with the rest used as a training set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=.25,
                                                    random_state=1)

clf = tree.DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)

# Visualize the decision tree,
tree.export_graphviz(clf,
                     out_file='tree.dot',
                     feature_names=list(X_train.columns),
                     class_names=['Did Not Churn', 'Churned'],
                     filled=True,
                     rounded=True,
                     special_characters=True,
                     proportion=True)

# Use the fitted model to test the unknown data points
y_prediction = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_prediction)
print(accuracy)
