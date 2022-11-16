#Kmeans Clustering 

#Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
df = pd.read_csv('sales_data_sample.csv', encoding = 'unicode_escape') #Reading the csv fi
df.head()
# //2)
df = pd.read_csv('sales_data_sample.csv',  encoding = 'unicode_escape') #Reading the csv file.
pd.set_option('display.max_columns', None)
# df.dtypes()
df.head()
# //3)
#Removing the coloumns which dont add value for the analysis.
to_drop = ['PHONE','ADDRESSLINE1','ADDRESSLINE2','CITY','STATE','POSTALCODE','TERRITORY']
df = df.drop(to_drop, axis=1)
df.head()
# //4)
df.nunique() #Checking unique values
# //5)
df.isnull().sum()

# //6)
#Removing the coloumns which dont add value for the analysis.
to_drop = ['PHONE','ADDRESSLINE1','ADDRESSLINE2','CITY','STATE','POSTALCODE','TERRITORY','CONTACTLASTNAME','CONTACTFIRSTNAME','CUSTOMERNAME','ORDERNUMBER','QTR_ID','ORDERDATE']
df = df.drop(to_drop, axis=1)
df.head()
# //7)
#Encodning Categorical Variables for easier processing.
status_dict = {'Shipped':1, 'Cancelled':2, 'On Hold':2, 'Disputed':2, 'In Process':0, 'Resolved':0}
df['STATUS'].replace(status_dict, inplace=True)
df['PRODUCTCODE'] = pd.Categorical(df['PRODUCTCODE']).codes
df = pd.get_dummies(data=df, columns=['PRODUCTLINE', 'DEALSIZE', 'COUNTRY'])
df.dtypes
# //8)
#Using Heatmaps to find links between the data
plt.figure(figsize = (20, 20))
corr_matrix = df.iloc[:, :10].corr()
sns.heatmap(corr_matrix, annot=True);
# //9)
#Finding correlation between variables using pairplots
fig = px.scatter_matrix(df, dimensions=df.columns[:8], color='MONTH_ID') #Fill color by months
fig.update_layout(title_text='Sales Data', width=1100, height=1100)
fig.show()
# //10)
# Scale the data
std = StandardScaler()
sdf = std.fit_transform(df)
wcss = []
for i in range(1,15):
    km = KMeans(n_clusters=i)
    km.fit(sdf)
    wcss.append(km.inertia_) # intertia is the Sum of squared distances of samples to their closest cluster center (WCSS)

plt.plot(wcss, marker='+', linestyle='--')
plt.title('The Elbow Method (Finding right number of clusters)')
plt.xlabel('Number of CLusters')
plt.ylabel('WCSS')
plt.show()
# //11)
#Applying k-means with 5 clusters as the elbow seems to form at 5 clusters
km = KMeans(n_clusters=5, random_state=1)
km.fit(sdf)
cluster_labels = km.labels_
df = df.assign(Cluster=cluster_labels)
df.head()