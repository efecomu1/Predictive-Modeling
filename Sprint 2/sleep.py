"""
Sprint 2 - Sleep Deliverable
Efe Comu
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm

df = pd.read_csv('Data/Sleep.csv')
print(df.head())

print(df.columns)
## Q1-Variables
# Animal: Categorical
# Body, Brain, SWsleep, Parasleep, Totsleep, Life, Gest, Pred, Exposure, Danger: all quantitative

## Q2
min = df['Totsleep'].min()
print(min)
max = df['Totsleep'].max()
print(max)
median = df['Totsleep'].median()
print(median)
avg = (df['Totsleep'].sum()) / df['Totsleep'].count()
print(avg)

# The Totsleep data ranges from 2.9 hours a day all the way up to 19.9 hours a day depending on the mammal species.
# The dataset includes 42 species, with a total sleep average of 10.6 hours a day. Median value for the total sleep data is 9.8 hours a day.
# Even though mean and median values are similar, the mean is still a bit higher than the median, causing the total sleep data distribution to be right-skewed.

## Q3
sns.scatterplot(df, x='Parasleep', y='SWsleep')
plt.xlabel('Non-dreaming sleep time')
plt.ylabel('Dreaming sleep time')
plt.savefig('sleep.png', bbox_inches='tight')

# Even though we can't guarantee this statement for every mammal species, animals who spend more time in non-dreaming sleep are also tend to spend more time in dreaming sleep.

## Q4
# The appropriate method to model the relationship between Parasleep and SWsleep is linear regression.
# This is because both of these variables are quantitative, they show a weak / moderate (just based on the first visual observation) linear relationship, and they have no outliers that will heavily shift the regression graph

# Q5
plt.figure()
sns.regplot(df, x='Parasleep', y='SWsleep', ci=True)
plt.savefig('sleep_reg.png', bbox_inches='tight')

# Q6
corr = df.corr()
print(corr['SWsleep'])

Y = df['SWsleep']
X = df['Parasleep']
X = sm.add_constant(
  X)  # <-- sm requires calling add_constant to add an intercept
mod = sm.OLS(Y, X)
res = mod.fit()
print(res.params)
print(res.summary())

# SWsleep - Parasleep correlation: 0.5182
# These two variables have a moderate positive correlation, and it means that holding everything else constant, as one of the variables move into one direction, the other variable tends to move in the same direction, getting affected moderately.
# Rsquared = 0.269
# The RSquare (coefficient of determination) value suggests that 26.9% of the changes in dreaming sleep could be explained directly by non-dreaming sleep.

# Q7
# Formula of the line based on the regression summary: y = 6.0221 + 1.4320x
# 5 hours of non dreaming sleep: x = 5

ans = 6.0221 + 1.4320 * 5
print(ans)

# Q8
diff = df['SWsleep'] - df['Parasleep']
print(diff)
