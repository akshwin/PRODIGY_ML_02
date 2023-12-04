# Customer Segmentation using K-Means Clustering

## Overview

This repository contains code for performing customer segmentation using the K-Means clustering algorithm which is done as a part of internship at Bharat Intern . Customer segmentation is a common technique in marketing and data analysis that involves grouping customers based on similar characteristics, such as age, income, and spending behavior.

## Libraries Used

```python
# Importing Libraries
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.cluster import KMeans
from scipy.stats import zscore
import warnings 
warnings.simplefilter('ignore')
```

## Steps Followed

1. **Importing Libraries**:

```python
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.cluster import KMeans
from scipy.stats import zscore
import warnings 
warnings.simplefilter('ignore')
```

2. **Loading the Dataset**:

```python
# Loading the Dataset
df=pd.read_csv("Mall_Customers.csv")
```

3. **Exploratory Data Analysis (EDA)**:

```python
# Display the first few rows of the dataset
df.head()

# Check the shape and data types of the columns
df.shape
df.dtypes

# Check for missing values
df.isnull().sum()

# Check for duplicate rows
df.duplicated().sum()
```

4. **Model Building and Visualization**:

```python
# Drop the 'Gender' column
df = df.drop('Gender', axis=1)

# Scale the numerical features using z-score normalization
dfScaled = df.apply(zscore)

# Visualize the pairwise relationships between features using a pairplot
sns.pairplot(dfScaled, diag_kind='kde')
```

5. **K-Means Clustering**:

```python
# The elbow method 
clusters = range(1, 100)
meanDistortion = []

for k in clusters:
    model = KMeans(n_clusters=k)
    model.fit(dfScaled)
    prediction = model.predict(dfScaled)
    meanDistortion.append(sum(np.min(cdist(dfScaled, model.cluster_centers_, 'euclidean'), axis=1)) / dfScaled.shape[0])

# Visualize the elbow method results
plt.plot(clusters, meanDistortion, 'bx-')
plt.xlabel('k')
plt.ylabel('Mean distortion')
plt.title("Selecting k with elbow method")
```

6. **Model Building**:

```python
# Build the final K-Means clustering model
final_model = KMeans(n_clusters=4, init='k-means++', max_iter=300, random_state=None)
final_model.fit(dfScaled)
prediction = final_model.predict(dfScaled)

# Append the cluster information to the original dataset
df['Clusters'] = prediction
```

7. **Results and Analysis**:

```python
# Display the count of data points in each cluster
df['Clusters'].value_counts()

# Display specific information for each cluster
df[df['Clusters'] == 0]
df[df['Clusters'] == 1]
df[df['Clusters'] == 2]
df[df['Clusters'] == 3]
```

8. **Silhouette Score Calculation**:

```python
# Calculate the silhouette score
from sklearn.metrics import silhouette_score
silhouette_score_average = silhouette_score(dfScaled, final_model.predict(dfScaled))
print(silhouette_score_average)
```

## Conclusion

The K-Means clustering model successfully segmented customers into distinct clusters based on their age, annual income, and spending score. The silhouette score of 0.41358360614845524 indicates a relatively good separation between clusters.
