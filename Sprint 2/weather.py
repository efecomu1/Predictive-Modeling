"""
Analyze the Australian weather data set
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('Data/weatherAUS.csv')
print(df.head())

print(df.columns)

df['RainToday_Num'] = np.where(df['RainToday'] == 'No', 0, 1)
df['RainTomorrow_Num'] = np.where(df['RainTomorrow'] == 'No', 0, 1)

corr = df.corr()
print(corr['RainTomorrow_Num'])

## Answer: Cloud3pm and Humidity3pm have
# the most positive correlation with RainTomorrow_Num.
# Sunshine has the most negative correlation,
# whereas Temp9am has the correlation closest to 0.

print(corr['Humidity3pm'])

## Answer: With Humidity3pm, Humidity9am has the most positive correlation, which makes sense.
# Also, the variable that has the most negative correlation is Sunshine, which is a little bit controversial in my opinion, but again,
# "correlation doesn't mean causation"

log_reg = smf.logit('RainTomorrow_Num ~ RainToday_Num', data=df).fit()
print(log_reg.summary())

print(log_reg.pred_table(.5))

log_reg = smf.logit('RainTomorrow_Num ~ RainToday_Num + Humidity3pm',
                    data=df).fit()
print(log_reg.summary())

print(log_reg.pred_table(.5))

## Answer: According to the confusion matrix of the second logistic regression,
# true negatives = 101825
# true positives = 13557
# false negatives = 19726
# false positives = 5845
#
# Overall accuracy = (101825 + 13557) / (101825 + 13557 + 19726 + 5845)
# Overall accuracy = 115382 / 140953
# = 0.8186 --> 81.9%
# Adding a new variable increased the accuracy of the model.
