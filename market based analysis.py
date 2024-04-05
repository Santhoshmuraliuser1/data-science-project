import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from apyori import apriori

# Load the dataset
dataset = pd.read_csv('groceries_dataset.csv')
print('Dimensions of dataset are :', dataset.shape)

# Drop the 'Member_number' column
dataset = dataset.drop(columns='Member_number')

# Remove 'bags' from itemDescription
dataset = dataset[dataset['itemDescription'] != 'bags']

# Convert Date to datetime
dataset['Date'] = pd.to_datetime(dataset['Date'], format='%d-%m-%Y')

# Aggregate all the items sold on the same date into a single column
dataset = dataset.groupby('Date')['itemDescription'].apply(list).reset_index()

# Generate transactions
transactions = []
for indexer in range(len(dataset)):
    transactions.append(dataset['itemDescription'].iloc[indexer])

# Apply Apriori algorithm
rules = apriori(transactions=transactions, min_support=0.00412087912, min_confidence=0.6, min_lift=1.9, min_length=2, max_length=2)

# Convert results to list
results = list(rules)

# Putting the results into a Pandas DataFrame
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))

results_df = pd.DataFrame(inspect(results), columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

# Display the results
print(results_df)

# Set the style of seaborn
sns.set_style("whitegrid")

# Plot the association rules
plt.figure(figsize=(12, 8))

# Bar plot for support, confidence, and lift
plt.subplot(2, 1, 1)
sns.barplot(x='Support', y='Left Hand Side', data=results_df, color='skyblue', label='Support')
sns.barplot(x='Confidence', y='Left Hand Side', data=results_df, color='salmon', label='Confidence')
sns.barplot(x='Lift', y='Left Hand Side', data=results_df, color='lightgreen', label='Lift')
plt.title('Association Rules')
plt.xlabel('Metrics')
plt.ylabel('Left Hand Side')
plt.legend()

# Scatter plot for support vs. confidence vs. lift
plt.subplot(2, 1, 2)
sns.scatterplot(x='Support', y='Confidence', size='Lift', data=results_df, sizes=(20, 200))
plt.title('Support vs. Confidence vs. Lift')
plt.xlabel('Support')
plt.ylabel('Confidence')

plt.tight_layout()
plt.show()

