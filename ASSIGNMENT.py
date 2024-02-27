#!/usr/bin/env python
# coding: utf-8

# In[92]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np


# In[93]:


books_df = pd.read_csv('Books_df.csv')


# In[94]:


books_df.drop(columns=['Unnamed: 0'], inplace=True)


# In[95]:


books_df['Price'] = books_df['Price'].str.replace('₹', '').str.replace(',', '').astype(float)


# In[96]:


missing_values = books_df.isnull().sum()


# In[97]:


dtypes_after_cleanup = books_df.dtypes


# In[98]:


books_df['Author'].fillna('Unknown', inplace=True)


# In[99]:


numerical_analysis = books_df[['Price', 'Rating', 'No. of People rated']].describe()


# In[100]:


numerical_analysis


# In[32]:


def detect_outliers_iqr(data, feature):
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
    return outliers, lower_bound, upper_bound


# In[33]:


price_outliers, price_lb, price_ub = detect_outliers_iqr(books_df, 'Price')


# In[34]:


rating_outliers, rating_lb, rating_ub = detect_outliers_iqr(books_df, 'No. of People rated')


# In[35]:


outliers_summary = {
    'Price': {
        'Number of Outliers': price_outliers.shape[0],
        'Lower Bound': price_lb,
        'Upper Bound': price_ub
    },
    'No. of People rated': {
        'Number of Outliers': rating_outliers.shape[0],
        'Lower Bound': rating_lb,
        'Upper Bound': rating_ub
    }
}

outliers_summary


# In[36]:


outliers_summary = {
    'Price': {
        'Number of Outliers': price_outliers.shape[0],
        'Lower Bound': price_lb,
        'Upper Bound': price_ub
    },
    'No. of People rated': {
        'Number of Outliers': rating_outliers.shape[0],
        'Lower Bound': rating_lb,
        'Upper Bound': rating_ub
    }
}

outliers_summary


# In[37]:


outliers_summary = {
    'Price': {
        'Number of Outliers': price_outliers.shape[0],
        'Lower Bound': price_lb,
        'Upper Bound': price_ub
    },
    'No. of People rated': {
        'Number of Outliers': rating_outliers.shape[0],
        'Lower Bound': rating_lb,
        'Upper Bound': rating_ub
    }
}

outliers_summary


# In[38]:


sns.set_style("whitegrid")


# In[39]:


plt.figure(figsize=(12, 6), dpi=200)
sns.histplot(books_df['Price'], bins=50, kde=True)
plt.title('Distribution of Price')
plt.xlabel('Price (₹)')
plt.ylabel('Frequency')
plt.show()


# In[40]:


plt.figure(figsize=(12, 6), dpi=200)
sns.histplot(books_df['Rating'], bins=30, kde=True)
plt.title('Distribution of Rating')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()


# In[41]:


plt.figure(figsize=(12, 6), dpi=200)
sns.histplot(books_df['No. of People rated'], bins=50, kde=True)
plt.title('Distribution of No. of People rated')
plt.xlabel('No. of People rated')
plt.ylabel('Frequency')
plt.xlim(0, books_df['No. of People rated'].quantile(0.95))


# In[42]:


plt.figure(figsize=(12, 8), dpi=200)
books_df['Main Genre'].value_counts().plot(kind='barh')
plt.title('Distribution of Books Across Main Genres')
plt.xlabel('Number of Books')
plt.ylabel('Main Genre')
plt.show()


# In[43]:


min_ratings_threshold = 1000


# In[44]:


author_stats = books_df.groupby('Author').agg({
    'Rating': 'mean',
    'Price': 'mean',
    'No. of People rated': ['sum', 'mean']
}).reset_index()


# In[45]:


author_stats.columns = ['Author', 'Avg Rating', 'Avg Price', 'Total Ratings', 'Avg Ratings per Book']


# In[46]:


best_rated_authors = author_stats[author_stats['Avg Ratings per Book'] > min_ratings_threshold].sort_values(by='Avg Rating', ascending=False).head(10)


# In[47]:


most_expensive_authors = author_stats.sort_values(by='Avg Price', ascending=False).head(10)


# In[48]:


most_rated_authors = author_stats.sort_values(by='Total Ratings', ascending=False).head(10)

best_rated_authors, most_expensive_authors, most_rated_authors


# In[49]:


plt.figure(figsize=(18, 14), dpi=200)


# In[50]:


plt.subplot(3, 1, 1)
sns.barplot(x='Avg Rating', y='Author', data=best_rated_authors, palette='cool')
plt.title('Top 10 Best Rated Authors')
plt.xlabel('Average Rating')
plt.ylabel('Author')


# In[51]:


plt.subplot(3, 1, 2)
sns.barplot(x='Avg Price', y='Author', data=most_expensive_authors, palette='autumn')
plt.title('Top 10 Most Expensive Authors')
plt.xlabel('Average Price (₹)')
plt.ylabel('Author')


# In[52]:


plt.subplot(3, 1, 3)
sns.barplot(x='Total Ratings', y='Author', data=most_rated_authors, palette='spring')
plt.title('Top 10 Most Rated Authors')
plt.xlabel('Total Ratings')
plt.ylabel('Author')

plt.tight_layout()
plt.show()


# In[53]:


highly_rated_books = books_df[books_df['Rating'] >= 4.5]


# In[54]:


ratings_threshold = highly_rated_books['No. of People rated'].quantile(0.25)


# In[55]:


hidden_gems = highly_rated_books[highly_rated_books['No. of People rated'] <= ratings_threshold]


# In[56]:


hidden_gems_sorted = hidden_gems.sort_values(by=['Rating', 'Price'], ascending=[False, True])


# In[57]:


hidden_gems_sorted.head(10)


# In[58]:


plt.figure(figsize=(12, 8), dpi=200)
sns.scatterplot(x='Price', y='Rating', data=books_df, alpha=0.5)
plt.title('Price vs. Rating of Books')
plt.xlabel('Price (₹)')
plt.ylabel('Rating')
plt.xscale('log')  # Using a logarithmic scale for better visualization of a wide range of prices
plt.grid(True, which="both", ls="--")
plt.show()


# In[59]:


books_per_author = books_df['Author'].value_counts().reset_index()
books_per_author.columns = ['Author', 'Number of Books']



# In[60]:


top_authors = books_per_author.head(10)


# In[61]:


plt.figure(figsize=(12, 8), dpi=200)
sns.barplot(x='Number of Books', y='Author', data=top_authors, palette='viridis')
plt.title('Top 10 Most Prolific Authors')
plt.xlabel('Number of Books')
plt.ylabel('Author')
plt.show()


# In[72]:


plt.figure(figsize=(14, 10), dpi=200)
sns.boxplot(y='Main Genre', x='Rating', data=books_df, palette="coolwarm")
plt.title('Ratings Distribution by Main Genre')
plt.xlabel('Rating')
plt.ylabel('Main Genre')
plt.xlim(0, 5)  
plt.show()


# In[73]:


significant_ratings_threshold = books_df['No. of People rated'].quantile(0.50)
filtered_books = books_df[books_df['No. of People rated'] >= significant_ratings_threshold]


# In[74]:


top_books_per_genre = filtered_books.loc[filtered_books.groupby('Main Genre')['Rating'].idxmax()]


# In[75]:


top_books_display = top_books_per_genre[['Title', 'Author', 'Main Genre', 'Rating', 'No. of People rated']]


# In[76]:


top_books_display_sorted = top_books_display.sort_values(by='Main Genre')


# In[77]:


top_books_display_sorted


# In[78]:


features = books_df[['Price', 'Rating', 'No. of People rated']]


# In[79]:


scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# In[80]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(features_scaled)
    wcss.append(kmeans.inertia_)


# In[81]:


plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # Within cluster sum of squares
plt.xticks(np.arange(1, 11, 1))
plt.grid(True)
plt.show()


# In[82]:


kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_labels = kmeans.fit_predict(features_scaled)


# In[83]:


books_df['Cluster'] = cluster_labels


# In[84]:


from sklearn.decomposition import PCA


# In[85]:


pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)


# In[86]:


plt.figure(figsize=(10, 8), dpi=200)
sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], hue=cluster_labels, palette='viridis', s=50, alpha=0.6)
plt.title('Books Clustering based on Price, Rating, and No. of People Rated (PCA-reduced to 2D)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()


# In[ ]:




