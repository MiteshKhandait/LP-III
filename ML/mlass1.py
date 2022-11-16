//1
#Importing required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
//2

df = pd.read_csv(r"C:\Users\Sachin Yadav\Desktop\mlcodeBE\uber.csv") #Reading CSV file
df.head()
df.info()
//3
df.dtypes #Checking Datatypes.
//4
df.isnull().sum() #Checking for Null Values.
//4
df.drop(['Unnamed: 0','key'],axis=1,inplace=True) #Dropping first coloumnn as it is irrelevant.
# df.drop(['Unnamed: 0','key'],axis=1,inplace=True)
df.dropna(axis=0,inplace=True) #Dropping the rows with null values.
df.head()
//5
def haversine (lon_1, lon_2, lat_1, lat_2): #Function to find the distance using the coordinates
    lon_1, lon_2, lat_1, lat_2 = map(np.radians, [lon_1, lon_2, lat_1, lat_2]) #Converting Degrees to Radians
    diff_lon = lon_2 - lon_1
    diff_lat = lat_2 - lat_1
    distance = 2 * 6371 * np.arcsin(np.sqrt(np.sin(diff_lat/2.0)**2+np.cos(lat_1)*np.cos(lat_2)*np.sin(diff_lon/2.0)**2)) #Calculationg the Distance using Haversine Formula
    return distance

df['Distance']= haversine(df['pickup_longitude'],df['dropoff_longitude'],df['pickup_latitude'],df['dropoff_latitude'])
df['Distance'] = df['Distance'].astype(float).round(2) #Rounding-off to 2 decimals
df.head()
//6
#Plotting a scatter plot to check for outliers.
plt.scatter(df['Distance'], df['fare_amount'])
plt.xlabel("Distance")
plt.ylabel("fare_amount")

//7
#Dealing with Outliers via removing rows with too large Distance and 0 or lesser distance.
df.drop(df[df['Distance']>60].index, inplace=True)
# print(df.drop(df[df['Distance']>60].index, inplace=True))
df.drop(df[df['Distance']==0].index, inplace=True)
df.drop(df[df['Distance']<0].index, inplace=True)
#Dealing with Outliers via removing rows with 0 or lesser fare amounts.
df.drop(df[df['fare_amount']==0].index, inplace=True)
df.drop(df[df['fare_amount']<0].index, inplace=True)
#Uber fare 

#Dealing with Outliers via removing rows with non-plausible fare amounts and distance travelled.


df.drop(df[df['Distance']>100].index, inplace=True)
df.drop(df[df['fare_amount']>100].index, inplace=True)
df.drop(df[(df['fare_amount']>100) & (df['Distance']<1)].index, inplace = True )
df.drop(df[(df['fare_amount']<100) & (df['Distance']>100)].index, inplace = True )
#Plotting a Scatter Plot to check for any more outliers and also to show correlation between Fare Amount and Distance.
plt.scatter(df['Distance'], df['fare_amount'])
plt.xlabel("Distance")
plt.ylabel("fare_amount")
//8
#Preprocessing the Data Using Standard Scaler in range of -1 to 1
x = df['Distance'].values.reshape(-1, 1)        #Independent Variable
y = df['fare_amount'].values.reshape(-1, 1)     #Dependent Variable
std = StandardScaler()
Y = std.fit_transform(y)
X = std.fit_transform(x)
#Splitting the data into training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

//9
def apply_model(model): #Model to print the metrics of the various prediction models
    model.fit(X_train,Y_train)
    print("Training score = ",model.score(X_train,Y_train))
    print("Testing score = ",model.score(X_test,Y_test))
    print("Accuracy = ",model.score(X_test,Y_test))
    Y_pred = model.predict(X_test)
    print("Predicted values:\n",Y_pred)
    print("Mean Absolute Error =", metrics.mean_absolute_error(Y_test, Y_pred))
    print("Mean Squared Error =", metrics.mean_squared_error(Y_test, Y_pred))
    print("Root Mean Squared Error =", np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
//10
lr = LinearRegression()
apply_model(lr)
//11
#Random Forest Model
rf = RandomForestRegressor(n_estimators=100, random_state=10)
apply_model(rf)