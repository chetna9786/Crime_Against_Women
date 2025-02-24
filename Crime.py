import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import plotly.express as px
# Ignore all warnings
warnings.filterwarnings('ignore')


# Load the datasets
crimes_df = pd.read_csv('E:\women\CrimesOnWomenData.csv')
description_df = pd.read_csv('E:\women\description.csv')

# Display the first few rows of each dataset
print("CrimesOnWomenData.csv - First 5 Rows:")
print(crimes_df.head())

print("\nDescription.csv - First 5 Rows:")
print(description_df.head())

# Create a dictionary for column renaming
column_names = {
    'Rape': 'Rape Cases',
    'K&A': 'Kidnap and Assault',
    'DD': 'Dowry Deaths',
    'AoW': 'Assault on Women',
    'AoM': 'Assault on Minors',
    'DV': 'Domestic Violence',
    'WT': 'Witchcraft'
}

# Rename columns in the dataset
crimes_df.rename(columns=column_names, inplace=True)

# Check the renamed columns
print("\nRenamed Columns:")
print(crimes_df.columns)

# Drop the unnecessary columns
crimes_df_cleaned = crimes_df.drop(columns=['Unnamed: 0'])

# Check the cleaned DataFrame
print("\nCleaned Dataset Columns:")
print(crimes_df_cleaned.columns)
print("\nFirst 5 Rows of the Cleaned Dataset:")
print(crimes_df_cleaned.head())

# Dataset info
print("\nCleaned Dataset Info:")
crimes_df_cleaned.info()

# Summary statistics
print("\nSummary Statistics:")
print(crimes_df_cleaned.describe(include='all'))
print(crimes_df_cleaned.isnull().sum())

# Pivot the data for heatmap
heatmap_data = crimes_df_cleaned.pivot_table(values='Rape Cases', index='State', columns='Year', aggfunc='sum', fill_value=0)

plt.figure(figsize=(15, 10))
sns.heatmap(heatmap_data, cmap="YlGnBu", linecolor='white', linewidths=0.5)
plt.title('Heatmap of Rape Cases by State and Year')
plt.xlabel('Year')
plt.ylabel('State')
plt.show()

# Group by year and sum up all crime types
crime_trend = crimes_df_cleaned.groupby('Year').sum()

# Plotting the trend of different crimes over the years
plt.figure(figsize=(10, 8))
sns.lineplot(data=crime_trend)
plt.title('Trend of Crimes Against Women (2001-2021)')
plt.xlabel('Year')
plt.ylabel('Number of Crimes')
plt.xticks(rotation=45)
plt.legend(title='Crime Type', bbox_to_anchor=(1.05, 1), loc='upper right')
plt.show()

crimes_df_cleaned_eda = crimes_df_cleaned.drop(columns=['Year'])


# Total crimes by state
state_crime = crimes_df_cleaned_eda.groupby('State').sum().sort_values(by='Rape Cases', ascending=False)

# Top 10 states with the highest number of crimes
top_states = state_crime.head(10)

plt.figure(figsize=(12, 6))
top_states.plot(kind='bar', stacked=True)
plt.title('Top 10 States with the Highest Number of Crimes Against Women')
plt.xlabel('State')
plt.ylabel('Number of Crimes')
plt.xticks(rotation=45)
plt.show()

# Sum up all crimes to get a sense of distribution
crime_distribution =crimes_df_cleaned_eda.drop(['State'], axis=1).sum()

plt.figure(figsize=(10, 6))
crime_distribution.plot(kind='bar', color='teal')
plt.title('Distribution of Different Types of Crimes Against Women')
plt.xlabel('Crime Type')
plt.ylabel('Total Number of Crimes')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 8))
correlation_matrix = crimes_df_cleaned.drop(['State'], axis=1).corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Different Crimes')
plt.show()

# Calculate the proportion of each crime type within each state
state_crime_proportions = crimes_df_cleaned_eda.set_index('State').div(crimes_df_cleaned_eda.set_index('State').sum(axis=1), axis=0)

# Plot the proportions for the top 10 states
top_states = state_crime_proportions.head(10)
top_states.plot(kind='bar', stacked=True, figsize=(14, 8))
plt.title('Crime Proportions in Top 10 States')
plt.xlabel('State')
plt.ylabel('Proportion of Total Crimes')
plt.xticks(rotation=45)
plt.show()

# Calculate the mean number of each crime type for the top 10 states with the highest total crime numbers
top_states_mean_crime = crimes_df_cleaned_eda.groupby('State').mean().sort_values(by='Rape Cases', ascending=False).head(10)

# Plot the comparison
plt.figure(figsize=(14, 8))
top_states_mean_crime.plot(kind='bar', figsize=(14, 8))
plt.title('Average Number of Different Crime Types in Top 10 States')
plt.xlabel('State')
plt.ylabel('Average Number of Crimes')
plt.xticks(rotation=45)
plt.show()

import seaborn as sns

# Select top N states by total crime numbers
top_n_states = crimes_df_cleaned.groupby('State').sum().nlargest(6, 'Rape Cases').index

# Filter data for these states
filtered_df = crimes_df_cleaned[crimes_df_cleaned['State'].isin(top_n_states)]

# Create a facet grid to show trends over the years
g = sns.FacetGrid(filtered_df, col="State", col_wrap=3, height=4)
g.map(sns.lineplot, 'Year', 'Rape Cases')
g.set_titles("{col_name}")
g.set_axis_labels("Year", "Number of Rape Cases")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Rape Cases Trends in Top 6 States', fontsize=16)
plt.show()

# Melt the DataFrame to plot multiple crime types
melted_df = crimes_df_cleaned.melt(id_vars=['State'], var_name='Crime Type', value_name='Number of Crimes')

plt.figure(figsize=(14, 8))
sns.boxplot(x='Crime Type', y='Number of Crimes', data=melted_df)
plt.title('Distribution of Different Crimes Against Women Across States')
plt.xticks(rotation=45)
plt.show()

import plotly.express as px
import pandas as pd

# Create a scatter plot with states on the x-axis and a dummy y-axis (for visualization purposes)
fig = px.scatter(crimes_df_cleaned, 
                 x="State", 
                 y=[0]*len(crimes_df_cleaned),  # Dummy Y axis
                 size="Rape Cases", 
                 color="Rape Cases", 
                 hover_name="State", 
                 title="Rape Cases in India by State",
                 size_max=100,
                 color_continuous_scale=px.colors.sequential.Viridis)  # Change color scale here

fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))

# Increase plot width
fig.update_layout(yaxis=dict(visible=False), 
                  xaxis=dict(tickangle=45), 
                  showlegend=False,
                  width=1200)  # Adjust width here

fig.show()

