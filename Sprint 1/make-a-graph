"""
Make-a-graph problem
Sprint 1 
Efe Comu
"""

# Research Question: Not including the effect of inflation on the USD, what is the difference between the total earnings of the top 10 earner athletes in 1990 and 2020?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Forbes Richest Athletes 1990-2020.csv', delimiter=";")

print(df.head())

# 1990 top 10 earners
year_earnings1990 = df[df['Year'] == 1990]
print(year_earnings1990)
total_1990_beforeinf = year_earnings1990['earnings ($ million)'].sum()

## According to the CPI inflation calculator, between years 1990 and 2020, there was a cumulative price increase of 98.02% for USD
## Source: https://www.in2013dollars.com/us/inflation/1990?endYear=2020&amount=128000000

total_1990 = total_1990_beforeinf * 1.9802
print("%.2f" % total_1990)

# 2020 top 10 earners
year_earnings2020 = df[df['Year'] == 2020]
print(year_earnings2020)
total_2020 = year_earnings2020['earnings ($ million)'].sum()
print("%.2f" % total_2020)

plt.figure()
x_axis = ['1990', '2020']
y_axis = [total_1990, total_2020]
plt.bar(x_axis, y_axis, color=[
  'red',
  'green',
])
plt.title("Sum of earnings of top 10 athletes 1990 vs. 2020")
plt.ylabel("Earnings ($ million)")
plt.xlabel('Year')
plt.ylim([0, 900])
plt.savefig('earnings_bar.png', bbox_inches='tight')
plt.close()

## Make-a-graph report
#
# Due to the high increase of demand to sports mainly for advertising and entertainment purposes
# in the last centuries, earnings of the top athletes increased significantly as a result.
# This research focuses on finding how big this difference in the earnings of the top athletes is
# in the last 30 years. To accurately come to a conclusion, the effect of the inflation on the
# price value of the USD between the years 1990 - 2020 is taken into account and calculated before
# the comparison of the cumulative earnings of the top 10 athletes in both of these years. The key
# finding of the study suggests that the difference between the total of earnings of top 10 highest
# earned athletes in 1990 and 2020 is around 550 million U.S dollars, which accounts for more than
# the triple of the number in 1990.
# The dataset used for this study is a dataset from Kaggle named 'Forbes Highest Paid Athletes 1990 -
# 2020'. The dataset contains 300 entries, 10 for each year of the 30 years it represents. It also
# contains 8 fields, which are S.NO, Name, Nationality, Current Rank, Previous Year Rank, Sport,
# Year, andearnings ($ million). However, this research mainly focuses on the 2 fields, 'Year' and
# 'earnings', to efficiently lead to the answer to the research question.
# The results of this research are presented on a bar graph called 'earnings_bar'. The x-axis of
# the graph has the values 1990 and 2020, which are the years to be compared on the y-axis based on
# the cumulative top 10 earnings in those years. The y-axis values are the earnings, in million
# dollars, and the columns show the sum of the earnings of the top 10 earned athletes in 1990 and
# 2020. The increase in the size of the bar corresponds to a higher cumulative earning, as the
# y-axis value it matches with increases with it as well.
# The red column is the representation of the cumulative earnings of the top 10 athletes in 1990,
# and represents the value in million dollars after the effect of inflation (98.2%) is calculated.
# The green column represents the cumulative earnings of the top 10 athletes in 2020, and the
# difference between the two years is seen clearly on the bar graph. The red column represents a
# value of 253.47 million dollars, whereas the green column represents a value of 819.20 million
# dollars. This leads to the conclusion that between the years of 1990 - 2020, the cumulative
# earnings of top 10 highest earning athletes increased by 565.73 million dollars, which also is
# a 221.90% increase in the total earnings.
# As the research findings suggest, the increasing demand and advertising to the sports industry
# led to a significant increase in the earnings of the high-level professional athletes as well.
# 221.90% increase without the effect of inflation is a really big percentage jump that could
# definitely inspire a lot of teenagers to pursue their professional athlete career. Therefore,
# an increase in the competition in professional sports is expected in the following years.
#
# Efe Comu
#
# Dataset source: https://www.kaggle.com/datasets/parulpandey/forbes-highest-paid-athletes-19902019
