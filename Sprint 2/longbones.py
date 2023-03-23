"""
Analyze the longbones dataset, examining the
relationship between bone nitrogen content and
length of interment

This example illustrates linear regression and
some related features
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Open and view the dataset
#
# All the datasets are in the Data subdirectory
df = pd.read_csv('Data/Longbones.csv')
## print(df)

## sns.scatterplot(df, x='Nitro', y='Time')
#plt.xlabel('Nitrogen content (g per 100g of bone)')
#plt.ylabel('Length of interment (years)')
#plt.xlim(3, 4.5)
#plt.ylim(0, 100)
#plt.savefig('time_vs_nitro.png', bbox_inches='tight')

plt.figure()
sns.scatterplot(df, x='Nitro', y='Time', hue='Oil')
plt.xlabel('Nitrogen content (g per 100g of bone)')
plt.ylabel('Length of interment (years)')
plt.xlim(3, 4.5)
plt.ylim(0, 100)
plt.legend(['Contaminated with oil', 'No contamination'], loc='lower left')
plt.savefig('time_vs_nitro_with_oil.png', bbox_inches='tight')

# Remove the three contaminated data points
# by selecting only the points that have Oil == 0
#
# Here, Oil != 1 means "Oil not equal to 1"
df = df[df['Oil'] != 1]
#print(df)

corr = df.corr()
print(corr['Time'])

Y = df['Time']
X = df['Nitro']
X = sm.add_constant(
  X)  # <-- sm requires calling add_constant to add an intercept
mod = sm.OLS(Y, X)
res = mod.fit()
print(res.params)

print(res.summary())

plt.figure()
sns.regplot(df, x='Nitro', y='Time', ci=True)
plt.savefig('regplot.png', bbox_inches='tight')

plt.figure()
sns.residplot(df, x='Nitro', y='Time')
plt.savefig('residuals.png', bbox_inches='tight')

expected_y = 335.47 - 74.26 * 3.71
print(expected_y)
