"""
Sprint 2 - The Dream of Wearing Shorts Forever Deliverable
Efe Comu
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

## Q1
df = pd.read_csv('Data/weatherAUS.csv')
print(df.head())

## Q2
df['RainToday_Num'] = np.where(df['RainToday'] == 'No', 0, 1)

## Q3
logistic_reg = smf.logit(
  'RainToday_Num ~ WindDir9am + WindSpeed9am + Humidity9am + Pressure9am + Cloud9am + Temp9am',
  data=df).fit()

## Q4
print(logistic_reg.summary())

# Based on the logistic regression summary, the 9am variables that have the highest significance on RainToday_Num are Cloud9am and Humidity9am
# They have the highest coeff value and do not include 0 in their confidence intervals

## Q5

log_reg_1 = smf.logit(
  'RainToday_Num ~ Humidity9am + Sunshine + Cloud9am + MinTemp',
  data=df).fit()
print(log_reg_1.pred_table(.5))

## Q6
# just predicting "No" for every instance, could be done like it's done in the lab
df['RainTomorrow_Num'] = np.where(df['RainTomorrow'] == 'No', 0, 1)
log_reg = smf.logit('RainTomorrow_Num ~ RainToday_Num', data=df).fit()
print(log_reg.pred_table(.5))

# My model, done in Q5 including some of the significant variables in my opinion, has an overall accuracy of 80.90% predicting Rain_Today_Num.
#
# On the other hand, predicting "No" for every instance has an overall accuracy of 75.84%.
#
# Even though the difference is just above 5%, my model performs better.

## Q7
darwin = df[df['Location'] == 'Darwin']
log_reg_2 = smf.logit(
  'RainToday_Num ~ Humidity9am + Sunshine + Cloud9am + MinTemp',
  data=darwin).fit()
print(log_reg_2.pred_table(.5))

## print(log_reg_1.pred_table(.5))

uluru = df[df['Location'] == 'Uluru']
## uluru_rain = uluru[df['RainToday_Num'] == 1]

log_reg_3 = smf.logit('RainToday_Num ~ MinTemp + MaxTemp', data=uluru).fit()
print(log_reg_3.pred_table(.5))

## The model I built for the location Darwin uses variables Humidity9am, Sunshine, Cloud9am, and MinTemp to predict the variable RainToday_Num.
## The model has 2168 true negatives, 329 false negatives, 170 false positives, and 519 true positives, leading to an overall accuracy of 1 - (499 / 3186) = 84.34%
#
## The model I built for the location Uluru uses variables MaxTemp and MinTemp to predict the variable RainToday_Num.
## That model has 1389 true negatives, 120 false negatives, 12 false positives, and 18 true positives, leading to an overall accuracy of 1 - (132 / 1539) = 91.42%
