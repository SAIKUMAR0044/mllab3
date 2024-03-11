import pandas as pd


df = pd.DataFrame({'C': [220, 180, 250, 190], 'category': ['None', 'None', 'None', 'None']})
n = df.shape[0]

for i in range(n):
    if df.loc[i, 'C'] > 200:
        df.loc[i, 'category'] = "rich"
    else:
        df.loc[i, 'category'] = "poor"

print(df)

# Sample data for Wednesday in April
df1 = pd.DataFrame({'Price': [100, 120, 150, 90], 'Day': ['Wed', 'Wed', 'Wed', 'Wed'], 'Month': ['Apr', 'Apr', 'Apr', 'Apr'], 'Chg%': [5, -2, 8, -3]})

mean_D = df1['Price'].mean()

# Calculate the variance of the 'D' column
variance_D = df1['Price'].var()

print('Mean:', mean_D)
print('Variance:', variance_D)

# Select the rows where the day of the week is Wednesday
wednesday_df = df1[df1['Day'] == 'Wed']

# Calculate the mean of the 'Price' column for these rows
wednesday_mean = wednesday_df['Price'].mean()

# Calculate the mean of the 'Price' column for all rows (population mean)
population_mean = df1['Price'].mean()

print('Wednesday Mean:', wednesday_mean)
print('Population Mean:', population_mean)

April_df = df1[df1['Month'] == 'Apr']

April_mean = April_df['Price'].mean()

population_mean = df1['Price'].mean()

print('April Mean:', April_mean)
print('Population Mean:', population_mean)

l2 = list(map(lambda v: v < 0, df1['Chg%']))

# Store only the False values
l2_false = [value for value in l2 if value is False]

probability = (len(l2_false) / len(l2))*100

print(f'Probability: {probability}%')

l3 = list(map(lambda v: v > 0, wednesday_df['Chg%']))

l3_True = [value for value in l3 if value is True]

probability_wed = (len(l3_True) / len(l3))*100

conditional_prob = probability_wed / wednesday_df.shape[0]

print(f'profits on wednesday: {probability_wed}%')

print(f'conditional probability: {conditional_prob}%')

import seaborn as sns
import matplotlib.pyplot as plt

# Create a scatter plot of 'Chg%' data against the day of the week
sns.scatterplot(x='Day', y='Chg%', data=df1)

# Display the plot
plt.show()