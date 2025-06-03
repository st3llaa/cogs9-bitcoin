import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import ttest_ind


# Step 1: Import CPI data CSV (you need to download it from World Bank site first)
cpi_df = pd.read_csv('inflation.csv', skiprows=4)
  # adjust file name and skiprows as needed

# Step 2: Filter US data
us_cpi_df = cpi_df[cpi_df['Country Name'] == 'United States']

# Step 3: Process years and reduce columns to Year and Inflation Rate
# The World Bank CPI data usually has years as columns, so we need to melt it
us_cpi_melted = us_cpi_df.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
                              var_name='Year', value_name='Inflation Rate')

# Filter valid years (convert year to int and drop NaNs)
us_cpi_melted = us_cpi_melted[us_cpi_melted['Year'].str.isdigit()]
us_cpi_melted['Year'] = us_cpi_melted['Year'].astype(int)
us_cpi_melted = us_cpi_melted.dropna(subset=['Inflation Rate'])

# Keep only Year and Inflation Rate columns
us_cpi_final = us_cpi_melted[['Year', 'Inflation Rate']].reset_index(drop=True)


# Step 4: Import Bitcoin data CSV (downloaded from Kaggle)
bitcoin_df = pd.read_csv('Bitcoin.csv')  # adjust file name if needed

# Convert 'Timestamp' to datetime
bitcoin_df['Timestamp'] = pd.to_datetime(bitcoin_df['Timestamp'])

# Set datetime column as index
bitcoin_df.set_index('Timestamp', inplace=True)

# Resample to annual frequency and calculate the average closing price per year
bitcoin_annual = bitcoin_df['Close'].resample('Y').mean().reset_index()

# Extract the year and rename columns
bitcoin_annual['Year'] = bitcoin_annual['Timestamp'].dt.year
bitcoin_annual.rename(columns={'Close': 'Avg_Bitcoin_Price'}, inplace=True)

# Keep only the relevant columns
bitcoin_final = bitcoin_annual[['Year', 'Avg_Bitcoin_Price']]

# Calculate daily % return for Bitcoin
btc_df['btc_return'] = btc_df['Close'].pct_change()

# Merge BTC and CPI data on date or align CPI to announcement effect
# Assume CPI announcements affect the NEXT day's price
btc_df['next_day_return'] = btc_df['btc_return'].shift(-1)

# Merge CPI with Bitcoin by aligning the CPI date with next day's return
merged_df = pd.merge_asof(cpi_df.sort_values('date'), 
                          btc_df[['date', 'next_day_return', 'btc_return', 'Close']],
                          on='date', direction='forward')

# Create a binary label: 1 if next day return > 0, else 0
merged_df['price_up'] = (merged_df['next_day_return'] > 0).astype(int)

# Example CPI feature: CPI surprise (actual - expected)
if 'expected_cpi' in merged_df.columns:
    merged_df['cpi_surprise'] = merged_df['actual_cpi'] - merged_df['expected_cpi']

# Summary stats
print(merged_df[['actual_cpi', 'next_day_return', 'cpi_surprise']].describe())

# Correlation
sns.heatmap(merged_df.corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()

# Boxplot: next-day returns on CPI days
sns.boxplot(data=merged_df, x='price_up', y='next_day_return')
plt.title("Bitcoin Return vs. CPI Announcement Outcome")
plt.xlabel("Price Up Next Day")
plt.ylabel("Return")
plt.show()

# Split by outcome
returns_up = merged_df[merged_df['price_up'] == 1]['next_day_return']
returns_down = merged_df[merged_df['price_up'] == 0]['next_day_return']

# T-test
t_stat, p_val = ttest_ind(returns_up.dropna(), returns_down.dropna())
print(f"T-test result: t-stat={t_stat:.4f}, p-value={p_val:.4f}")


# Select features and target
features = ['actual_cpi', 'cpi_surprise']  # add more if available
X = merged_df[features].dropna()
y = merged_df.loc[X.index, 'price_up']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
