# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np 
import locale
import matplotlib.pyplot as plt
import plotly.express as px  # Add this import
from datetime import datetime
from dateutil.relativedelta import relativedelta
import random
import plotly.graph_objects as go
import math
# import time


st.write('paul')
maximum_number_of_rows_of_data_to_use = 802

# Streamlit slider for income goal
income_goal = st.slider("Select your income goal:", 1000, 100000, step=100)
annuity_rate = st.number_input("Enter Annuity Rate:", min_value= 3, max_value=10, value=6, step=.1)

# Display the selected income goal
st.write(f"Your selected income goal is: ${income_goal}")

# Streamlit slider for number of years
number_of_years = st.slider("Select the number of years:", 1, 35,30, step=1)

# Use number_of_years for the maximum_number_of_years_in_plan
maximum_number_of_years_in_plan = number_of_years

# Create a DataFrame with the goal value for the selected number of years
goal_df = pd.DataFrame({'Year': range(1, maximum_number_of_years_in_plan + 1), 'Income Goal': [income_goal]*maximum_number_of_years_in_plan})

# Display the DataFrame
# st.write(goal_df)

# Values matrix cost at 0% Fee
matrix_cost_factors = [
    1.0000, 1.227416765, 1.2982, 1.292914924, 1.212937674, 1.1711, 1.197674184, 1.151232983, 1.0575, 1.074395942,
    0.986642879, 0.9195, 0.903393296, 0.884861038, 0.7921, 0.718634116, 0.611533665, 0.5508, 0.568348259, 0.558739825,
    0.5402, 0.471616516, 0.390915502, 0.3673, 0.377704059, 0.318953547, 0.2430, 0.194370295, 0.192592046, 0.1900,
    0.142090585, 0.138893469, 0.1223, 0.12762403, 0.103098357, 0.1000, 0.080258669, 0.081653593, 0.0880, 0.063018735,
    0.066015767
]


# Create a Pandas DataFrame for matrix cost factors
portfolio_funding_matrix_cost_all_columns = pd.DataFrame({'Cost': matrix_cost_factors})

# Adjust the length of portfolio_funding_matrix_cost_all_columns to match number_of_years
portfolio_funding_matrix_cost = portfolio_funding_matrix_cost_all_columns.head(maximum_number_of_years_in_plan)

# Multiply the cost factors by the income goal
multiplied_df = portfolio_funding_matrix_cost.copy()
multiplied_df['Cost'] *= goal_df['Income Goal'].values
total_cost = multiplied_df.sum()
# Output only the total cost value
total_cost_int = int(total_cost)


# Display the resulting DataFrame
# st.write(multiplied_df)



# Pull in the end_val.xlsx spreadsheet to get ending value data

# Read the Excel file with no headers
end_val_df = pd.read_excel('end_val.xlsx', header=None)


# 1. Now get the Ending Value Array which is the net_portfolio_goal (transposed) * a resizing of the end_val_df

resized_end_val_df = end_val_df.iloc[:maximum_number_of_rows_of_data_to_use, :maximum_number_of_years_in_plan]
# st.write(resized_end_val_df)



# Extract the row as a Series for broadcasting
cost_series = multiplied_df.iloc[0]

# Transpose multiplied_df to convert it into a single row
transposed_multiplied_df = multiplied_df.T

# Remove the 'Cost' header to align with resized_end_val_df column index
transposed_multiplied_df.columns = resized_end_val_df.columns

# Perform element-wise multiplication
# resized_end_val_df columns are multiplied by the corresponding value in transposed_multiplied_df
elementwise_product_df = resized_end_val_df.mul(income_goal)

# Display the resulting DataFrame
# st.write(elementwise_product_df)

# Load the cpi_end_val.xlsx worksheet
cpi_end_val_df = pd.read_excel('cpi_end_val.xlsx', header=None)

# Transpose income_goal_series and convert it to a DataFrame
income_goal_series = goal_df['Income Goal']
income_goal_df = pd.DataFrame({'Income Goal': income_goal_series}).T

# Ensure both DataFrames have the same number of columns (equal to number_of_years)
number_of_columns = number_of_years

# Resize income_goal_df and cpi_end_val_df to have the same number of columns
income_goal_df = income_goal_df.iloc[:, :number_of_columns]
cpi_end_val_df = cpi_end_val_df.iloc[:, :number_of_columns]

# Multiply cpi_end_val_df by income_goal_df element-wise
# This gives us each years ending real value of the income stream
result_df = cpi_end_val_df * income_goal_df.values

# Display the resulting DataFrame
# st.write(result_df)

# Calculate mean, median, and minimum for result_df
# This is the mean, median, and minimum for of each row of data in sourceresult_df of each row of data

result_mean = result_df.mean().mean()  # Calculate the mean of all elements
result_median = result_df.median().median()  # Calculate the median of all elements
result_min = result_df.min().min()  # Calculate the minimum of all elements

# Calculate mean, median, and minimum for elementwise_product_df
elementwise_mean = elementwise_product_df.mean().mean()  # Calculate the mean of all elements
elementwise_median = elementwise_product_df.median().median()  # Calculate the median of all elements
elementwise_min = elementwise_product_df.min().min()  # Calculate the minimum of all elements

annuity_rate = .0621
cost_of_annuity = (income_goal) / annuity_rate
cost_of_annuity_real = (income_goal / result_median) *  cost_of_annuity

# Output the results using st.write
st.markdown("# Results for Annuity Income Stream:")
st.write(f'Cost of the Annuity: ${cost_of_annuity:,.0f}')
st.write(f'Average Spending Value of the Annuity: ${result_mean:,.0f}')
st.write(f'Median Spending Value of the Annuity: ${result_median:,.0f}')
st.write(f'Lowest Spending Value of the Annuity: ${result_min:,.0f}')
st.write(f'Actual Cost to Get Real Income Stream Gauarantee: ${cost_of_annuity_real:,.0f}')



st.markdown("# Results for Iris:")
st.write(f'Total Cost to fund with Iris: ${total_cost_int:,.0f}')
st.write(f'Average Spending with Iris: ${elementwise_mean:,.0f}')
st.write(f'Median Spending with Iris: ${elementwise_median:,.0f}')
st.write(f'Minimum Spending with Iris: ${elementwise_min:,.0f}')


# now for raw inflation data

# Assuming resized_end_val_df is your DataFrame
# Select the first 802 rows of the 30th column and calculate the mean


inflation_mean_value = 1 / cpi_end_val_df.iloc[:maximum_number_of_rows_of_data_to_use, number_of_years - 1].mean()
inflation_median_value = 1 / cpi_end_val_df.iloc[:maximum_number_of_rows_of_data_to_use, number_of_years - 1].median()
inflation_min_value = 1 / cpi_end_val_df.iloc[:maximum_number_of_rows_of_data_to_use, number_of_years - 1].min()


formatted_mean = "{:.2f}".format(inflation_mean_value * 1)
formatted_median = "{:.2f}".format(inflation_median_value * 1)
formatted_min = "{:.2f}".format(inflation_min_value * 1)

# Format these numbers as currency without decimal places and display
st.write(f'Average Cost Increase Factor: {inflation_mean_value:,.2f}')
st.write(f'Median Cost Increase Factor: {inflation_median_value:,.2f}')
st.write(f'Minimum Cost Increase Factor: {inflation_min_value:,.2f}')
