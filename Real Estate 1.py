#!/usr/bin/env python
# coding: utf-8

# ### Background 
# The Office of Policy and Management maintains a listing of all real estate sales with a sales price 
# of $2,000 or greater that occur between October 1 and September 30 of each year. For each sale 
# record, the file includes town, property address, date of sale, property type (residential, 
# apartment, commercial, industrial, or vacant land), sales price, and property assessment.

#    ### Problem Statement:
# * Exploring Property Assessment and Sales Data for Informed Decision-Making. 
# In our quest for informed decision-making in real estate, we are presented with a comprehensive 
# dataset encompassing various attributes related to property assessment and sales transactions. 
# 

# In[1]:


# import necessary libraries
import numpy as np
import pandas as pd

# for visuals
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

plt.style.use('ggplot')


# In[2]:


# import and read file
df = pd.read_csv(r"C:\Users\PC\Downloads\Real_Estate_Sales_2001-2020_GL.csv", encoding='utf-8')
df


# ## Data Inspection

# In[3]:


# . head() ==> to get the first five rows
df.head()


# In[4]:


# . tail() ==> to get the last five rows
df.tail()


# In[5]:


# shape of data
df.shape


# In[6]:


# info of the data
df.info()


# In[7]:


# .dtypes==> for the data types of the columns 
df.dtypes


# In[8]:


# check columns #Pandas methods. t0_list is to list data
df.columns.to_list()


# In[9]:


# check for missing values # .isna() # .isnull() ==> the return boolean mask
df.isnull()


# In[10]:


# checking for missing values
df.isnull().sum()


# In[11]:


# visualize missing values
plt.figure(figsize = (12, 6))
sns.heatmap(df.isnull(), cbar = True, cmap = 'coolwarm')
plt.title('Visualizing Missing Values')
plt.show()


# In[12]:


# check for negative values in 'Sale Amount'
negative_price = df[df['Sale Amount'] < 0]
(negative_price)


# In[13]:


# check for negative values in 'Sale Amount'
negative_price = df[df['Sales Ratio'] < 0]
(negative_price)


# In[14]:


# check for negative values in 'Sale Amount'
negative_price = df[df['Assessed Value'] < 0]
(negative_price)


# ### Data Preprocessing/Wrangling
# - We will handle the missing values for `Address`,`PropertyType`,`Residential Type`,`Non Use Code`,`Location`, `Date Recorded`, `OPM remarks` and `CustomerID` by using the imputing with mode method to replace the null values.
# 
# - We will convert the `DateRecorded` to a `datetime` data type
# 

# In[15]:


# check the columns that are categorical
cat_cols = df.select_dtypes(include = ['category', 'object']).columns.to_list()
cat_cols


# In[16]:


# check the columns that are numerical
num_cols = df.select_dtypes(include = ['float', 'int64']).columns.to_list()
num_cols


# In[17]:


# value counts for categorical columns
for column in cat_cols:
    print(df[column].value_counts())


# #### Data Manipulation and Data Validation

# In[18]:


# fill the missing values with its mode

df['Address'].fillna(df['Address'].mode()[0], inplace = True)

df['Property Type'].fillna(df['Property Type'].mode()[0], inplace = True)

df['Residential Type'].fillna(df['Residential Type'].mode()[0], inplace = True)

df['Non Use Code'].fillna(df['Non Use Code'].mode()[0], inplace = True)

df['Assessor Remarks'].fillna(df['Assessor Remarks'].mode()[0], inplace = True)

df['Location'].fillna(df['Location'].mode()[0], inplace = True)

df['Date Recorded'].fillna(df['Date Recorded'].mode()[0], inplace = True)

df['OPM remarks'].fillna(df['OPM remarks'].mode()[0], inplace = True)

# convert the 'Date Recorded' to Datetime
df['Date Recorded'] = pd.to_datetime(df['Date Recorded'])


# check the head of the column
df.head()


# In[19]:


# confirm data types
df.dtypes


# In[20]:


# confirm if there are still missing values
df.isnull().sum()


# ## Exploratory Data Analysis
# ##### Univariate Analysis
# - you are considering the distribution of a variable or feature and its visualization

# In[21]:


# statistical summary of the data
df.describe()


# The total number of records for each variable is 997,213. The mean value for Assessed Value, Sale Amount, and Sales Ratio is 279,143.70,391,151.20 and 10.45, respectively. The minimum year is 2001, while the maximum year is 2020. This information suggest that our dataset covers a span of nearly two decades of property listing. Also, the date range goes from April 5, 1999, to September 30, 2021. which suggest a broad timeframe for recorded transactions.
# 

# In[22]:


# distribution of numerical column
columns_to_visualize = ['Assessed Value', 'Sale Amount', 'Sales Ratio']

# Create a figure and axes object
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Plot histograms for each numerical column
axs[0].hist(df['Assessed Value'], bins=20, color='blue')
axs[0].set_title('Assessed Value')
axs[0].set_xlabel('Value')
axs[0].set_ylabel('Frequency')

axs[1].hist(df['Sale Amount'], bins=20, color='green')
axs[1].set_title('Sale Amount')
axs[1].set_xlabel('Value')
axs[1].set_ylabel('Frequency')

axs[2].hist(df['Sales Ratio'], bins=20, color='orange')
axs[2].set_title('Sales Ratio')
axs[2].set_xlabel('Value')
axs[2].set_ylabel('Frequency')

# Adjust spacing between subplots if needed
plt.tight_layout()

# Display the chart
plt.show()


# From the graph above, most properties have assessed values concentrated between 0 and 2 and the distribution appears to be positively skewed. Similar to the assessed value, there is concentration of values between 0 and 1 which suggest variability in sale amounts with some properties being sold for higher amounts compare to others. Lastly, the sales ratio is the ratio of the sale amount to the assessed value and is used to assess the fairness of property assessment.
# The distribution also appears to be right-skewed, with most properties having sales ratios below 1 which shows that the sale amount is lower than the assessed value. However, there are also some properties with sales ratios above 1, suggesting instances that sale amount exceeds the assessed value.

# In[23]:


# the number of unique property type
print(f"The number of unique Property type is {df['Property Type'].nunique()} \nThey are as follow: \n{df['Property Type'].unique()}")


# In[24]:


# the number of unique residential type
print(f"The number of unique Residential type is {df['Residential Type'].nunique()} \nThey are as follow: \n{df['Residential Type'].unique()}")


# In[25]:


# the number of unique town
print(f"The number of unique Town is {df['Town'].nunique()} \nThey are as follow: \n{df['Town'].unique()}")


# In[26]:


# top 5 customers with the most purchases
top5_customer = df['Serial Number'].value_counts().head()
top5_customer


# In[27]:


# Calculate the top 5 customers with the most purchases
top5_customer = df['Serial Number'].value_counts().head()

# Create the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(top5_customer.index, top5_customer.values, color='skyblue')

# Add data labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center')

# Add labels and title
plt.xlabel('Customer Serial Number')
plt.ylabel('Number of Purchases')
plt.title('Top 5 Customers with the Most Purchases')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Show the plot
plt.tight_layout()
plt.show()


# The graph above shows the top 5 customers contributing to the most purchase and we can see similar purchase counts ranging from 171 to 172 purchases each. This consistency suggest that these customers have a regular buying pattern contributing significantly to sales over time.

# In[28]:


# Top 10 Town with the most purchases
top10_countries = df['Town'].value_counts()[:10]
top10_countries


# In[29]:


# Top 10 Towns with the most purchases
top10_towns = df['Town'].value_counts()[:10]

# Plot horizontal bar chart
plt.figure(figsize=(10, 6))
top10_towns.plot(kind='barh', color='skyblue')
plt.xlabel('Number of Purchases')
plt.ylabel('Town')
plt.title('Top 10 Towns with the Most Purchases')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest count at the top
plt.show()


# From the the graph above, Bridgeport tops the list with the highest number of purchases which shows that Bridgeport is a significant hub for property transactions. Stamford follows closely behind Bridgeport, indicating substantial purchasing activity in this town as well. Waterbury ranks third in terms of the number of purchases even despite being smaller than Bridgeport and Stamford, Waterbury still exhibits significant real estate activity. It could be as a result of the town's affordability, location, and historical significance that has contribute to its attractiveness to property buyers. Norwalk follows closely behind Waterbury, indicating robust real estate activity in this coastal town. New Haven, home to Yale University and a rich cultural heritage, ranks fifth in the list. Danbury, West Hartford, Hartford, Milford, and Meriden:
# These towns also feature prominently in the top 10 list, indicating significant real estate activity in each.
# Location, affordability, amenities, and economic opportunities may influence property transactions in these areas.

# ### Properties with sales ratio significantly above and below 1

# In[30]:


# Define threshold for significant deviation
threshold = 1.1  # Adjust as needed based on your analysis

# Identify properties with sales ratios significantly above or below 1
over_assessed = df[df['Sales Ratio'] > threshold]
under_assessed = df[df['Sales Ratio'] < 1 / threshold]

# Print summary statistics or additional information about identified properties
print("Properties with sales ratios significantly above 1 (potential over-assessment):")
print(over_assessed)

print("\nProperties with sales ratios significantly below 1 (potential under-assessment):")
print(under_assessed)


# The above shows properties with sales ratios significantly above or below 1 and we can observe that
# Properties with sales ratios significantly above 1 (potential over-assessment):
# These properties have sales ratios greater than the threshold 1, suggesting that they might be over-assessed. which means that the assessed value of the property is higher than its actual sale amount, leading to potentially higher property taxes for the owners. For example, property with serial number 2020180 in Berlin has an assessed value of 234,200 and a sale amount of 130,000, resulting in a sales ratio of 1.8015, which is significantly above the threshold.
# Also for those less than 1
# Properties with sales ratios significantly below 1 (potential under-assessment):
# These properties have sales ratios less than the inverse of the threshold 1, indicating that they might be under-assessed. Which means that the assessed value of the property is lower than its actual sale amount, potentially resulting in lower property taxes for the owners.
# For example, property with serial number 2020348 in Ansonia has an assessed value of 150,500 and a sale amount of 325,000, resulting in a sales ratio of 0.463, which is significantly below the threshold.
# As a result, this will help us identify properties that may need reassessment to ensure fairness and accuracy in property taxation.

# ### Bivariate Analysis
# - You are considering two features or variables and its visualization to understand the patterns, trends, and the measure of relationship between them.

# In[31]:


# Top 10 Town by Sales Ratio
top5_sales_town = df.groupby('Town')['Sales Ratio'].sum().sort_values(ascending = False)[:10]
top5_sales_town


# In[32]:


import plotly.express as px
import pandas as pd

# Create a DataFrame for the top 10 town by sales ratio
top5_sales_product = df.groupby('Town')['Sales Ratio'].sum().sort_values(ascending=False)[:10].reset_index()

# Create a bar chart with data labels
fig = px.bar(
    top5_sales_product,
    x='Sales Ratio',
    y='Town',
    text='Sales Ratio',  # This adds data labels
    labels={'Town': 'Town', 'Sales Ratio': 'Total Sales'},
    title='Top 10 Town by Sales Ratio (Bar Chart)',
)

# Customize the appearance of the bar chart
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_xaxes(title_text='')
fig.update_yaxes(categoryorder='total ascending')

# Show the customized bar chart
fig.show()


# The graph above shows that Salisbury has the highest total sales ratio, indicating that properties in Salisbury have generated the highest sales compared to their assessed values. Newtown follows Salisbury with a significant total sales ratio, suggesting strong property sales relative to assessed values in this town. Properties in New Fairfield have also generated substantial sales compared to their assessed values, ranking third in total sales ratio among the top 10 towns and with a notable total sales ratio, Westport indicates robust property sales relative to assessments. next is 
# East Hartford and similar to East Hartford, Brookfield ranks among the top towns with a considerable total sales ratio. Stamford appears in the list, signifying strong property sales relative to assessments in this town. Bethany's total sales ratio is notable, indicating significant property sales relative to assessed values. Properties in Guilford have generated substantial sales compared to their assessments, contributing to its total sales ratio. Rounding out the top 10, Beacon Falls has a considerable total sales ratio, indicating noteworthy property sales relative to assessments.
# Overall, these towns exhibit strong property sales relative to their assessed values, as reflected by their high total sales ratios. This data provides insights into the real estate market dynamics and assessment accuracy across different towns, informing stakeholders about areas with robust property sales activity.

# In[33]:


# Sales trend
sales_trend = df.groupby('List Year')['Sale Amount'].sum()
sales_trend


# In[34]:


#Assuming "List Year" is the column representing the year
sales_over_years = df.groupby('List Year')['Sale Amount'].sum().reset_index()

plt.figure(figsize=(10, 6))
plt.plot(sales_over_years['List Year'], sales_over_years['Sale Amount'], marker='o', linestyle='-', color='b')
plt.title('Real Estate Sales Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Sales Amount')
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.show()


# The line graph above shows a notable increase in sales from 2001 to 2004, with sales peaking in 2004. Subsequently, there was a slight decrease in sales from 2004 to 2005.
# Sales remained relatively stable from 2005 to 2009 before experiencing another slight decrease in 2009.
# From 2009 to 2020, there was a general upward trend in sales, with fluctuations in certain years.
# The highest sales amount was recorded in 2020, indicating a significant spike in real estate sales compared to previous years. The fluctuations in real estate sales can be influenced by factors such as, Economic Conditions like GDP growth, unemployment rates, interest rates among other growing factors. During periods of economic growth, people may feel more confident in investing in real estate, leading to an increase in sales. Conversely, during economic downturns, people may be more cautious, resulting in a decrease in sales. Another factor could be Housing Market Supply and Demand: Fluctuations in the supply and demand of housing units can affect sales. For example, if there is an oversupply of housing units compared to demand, it can lead to a decrease in sales as sellers compete to attract buyers. On the other hand, a shortage of housing units can drive up demand and lead to an increase in sales. Changes in government policies relating to taxation, housing subsidies, mortgage rates, and regulations can also impact on real estate sales. For instance, government incentives such as tax credits for first-time homebuyers can stimulate demand and boost sales.
# 

# In[35]:


# Scatter plot of sales amount vs. assessed value
plt.figure(figsize=(8, 6))
plt.scatter(df['Assessed Value'], df['Sale Amount'], alpha=0.5)
plt.title('Sale Amount vs. Assessed Value')
plt.xlabel('Assessed Value')
plt.ylabel('Sale Amount')
plt.grid(True)
plt.show()

# Calculate Pearson correlation coefficient
correlation_coefficient = df['Assessed Value'].corr(df['Sale Amount'])
print(f"Pearson correlation coefficient: {correlation_coefficient}")


# The scatter plot visualizes the relationship between the assessed value and the sale amount of properties and each point represents a property, the Pearson correlation coefficient of 0.1109 confirms this to be a weak positive correlation with its position determined by its assessed value on the x-axis and its sale amount on the y-axis. The overall trend appears to be slightly positive, indicating a weak positive correlation between the assessed value and the sale amount. This means that as the assessed value of a property increases, there is a tendency for the sale amount to also increase, and vice versa.
# In this case, even though there is a positive relationship, its value being close to 0 suggests that the relationship is not strong. Therefore, while there may be some association between the assessed value and sale amount, other factors likely play a more significant role in determining the sale amount of a property.

# In[36]:


# Set style
sns.set(style="whitegrid")

# Create subplots for each property type
fig, axes = plt.subplots(3, 1, figsize=(12, 19))

# Plot violin plots for assessment values by property type
sns.violinplot(x='Property Type', y='Assessed Value', data=df, ax=axes[0])
axes[0].set_title('Distribution of Assessed Values by Property Type')
axes[0].set_ylabel('Assessed Value')

# Plot violin plots for sales amounts by property type
sns.violinplot(x='Property Type', y='Sale Amount', data=df, ax=axes[1])
axes[1].set_title('Distribution of Sales Amounts by Property Type')
axes[1].set_ylabel('Sale Amount')

# Plot violin plots for sales ratios by property type
sns.violinplot(x='Property Type', y='Sales Ratio', data=df, ax=axes[2])
axes[2].set_title('Distribution of Sales Ratios by Property Type')
axes[2].set_ylabel('Sales Ratio')


plt.show()


# The first violin plot shows the distribution of assessed values for each property type. It provides an overview of the range and variability of assessed values within each property type category. We can observe the spread of assessed values and any potential outliers across property types. The second violin plot depicts the distribution of sale amounts for each property type. It helps in understanding how sale amounts vary across different property types and highlights any significant differences or similarities in the sale amounts. The third violin plot illustrates the distribution of sales ratios for each property type. The sales ratio represents the ratio of the sale amount to the assessed value and is a measure of assessment accuracy. This plot helps identify patterns/trends in assessment accuracy across different property types.
# 

# In[37]:


# Calculate average assessment value and sale amount for each non-use code
non_use_code_stats = df.groupby('Non Use Code')[['Assessed Value', 'Sale Amount']].mean()

# Sort by average assessment value and sale amount
top20_assessed_value = non_use_code_stats['Assessed Value'].nlargest(20)
top20_sale_amount = non_use_code_stats['Sale Amount'].nlargest(20)

# Plot top 20 distribution of assessment values by non-use code
plt.figure(figsize=(12, 20))
top20_assessed_value.plot(kind='bar', color='skyblue')
plt.title('Top 20 Distribution of Assessment Values by Non-Use Code')
plt.xlabel('Non-Use Code')
plt.ylabel('Average Assessment Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot top 20 distribution of sale amounts by non-use code
plt.figure(figsize=(12, 20))
top20_sale_amount.plot(kind='bar', color='lightgreen')
plt.title('Top 20 Distribution of Sale Amounts by Non-Use Code')
plt.xlabel('Non-Use Code')
plt.ylabel('Average Sale Amount')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:


# Convert 'date recorded' column to datetime format
df['Date Recorded'] = pd.to_datetime(df['Date Recorded'])

# Extract year from 'date recorded' column
df['Year'] = df['Date Recorded'].dt.year

# Group by year and sum the sales amounts
total_sales_by_year = df.groupby('Year')['Sale Amount'].sum()

# Line plot
plt.figure(figsize=(10, 6))
total_sales_by_year.plot(kind='line', marker='o', color='skyblue')
plt.title('Total Sales Generated by Year')
plt.xlabel('Year')
plt.ylabel('Total Sales Amount')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()



# ## Multivariate Analysis
# - you are considering two or more features and its visualization to discover trends and patterns, and the measure of relationship between them.

# In[41]:


# Create a correlation matrix of the numerical columns
corr_matrix = df[['Assessed Value', 'Sale Amount', 'Sales Ratio']].corr()

# Create a customized correlation matrix using Plotly Express
fig = px.imshow(
    corr_matrix,
    x=['Assessed Value', 'Sale Amount', 'Sales Ratio'],
    y=['Assessed Value', 'Sale Amount', 'Sales Ratio'],
    title='Correlation Matrix of Numerical Columns',
)

# Customize the color scale and axis labels
fig.update_xaxes(title_text='Columns')
fig.update_yaxes(title_text='Columns')
fig.update_layout(coloraxis_showscale=False)  # Hide the color scale

# Add correlation values to the matrix
corr_values = np.around(corr_matrix.values, 2)  # Round the values to two decimal places
annotations = []
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        annotations.append(
            dict(
                x=corr_matrix.columns[i],
                y=corr_matrix.columns[j],
                text=str(corr_values[i, j]),
                showarrow=False,
            )
        )
fig.update_layout(
    annotations=annotations,
)

# Show the correlation matrix
fig.show()


# Assessed Value vs. Sale Amount: The correlation coefficient between Assessed Value and Sale Amount indicates a positive correlation of approximately 0.11. This suggests a weak positive linear relationship between the assessed value of properties and their sale amounts. While the correlation is positive, it is relatively low, indicating that changes in assessed values do not consistently correspond to proportional changes in sale amounts.
# Assessed Value vs. Sales Ratio: The correlation coefficient between Assessed Value and Sales Ratio is not directly relevant because Sales Ratio is a derived variable rather than an independent measure. Therefore, the correlation value is not meaningful for assessing the relationship between assessed values and sales ratios.
# Sale Amount vs. Sales Ratio: Similarly, the correlation coefficient between Sale Amount and Sales Ratio is not informative due to the nature of the sales ratio calculation. Sales Ratio is calculated based on Sale Amount and Assessed Value, so it inherently incorporates information from both variables. Therefore, the correlation value between Sale Amount and Sales Ratio is not meaningful for assessing their independent relationship.
# Overall, the correlation matrix provides insights into the linear relationships between numerical columns in the dataset. However, it's important to note that correlation does not imply causation, and additional analysis may be needed to understand the underlying factors influencing the observed correlations.

# In[ ]:




