#emails using binary calssification

# //1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

# //2
df=pd.read_csv(r"C:\Users\Sachin Yadav\Desktop\mlcodeBE\emails.csv") #Reading the Dataset
df.head()
# //3
df.drop(columns=['Email No.'], inplace=True) #Dropping Email No. as it is irrelevant.
df.head()

# //3
#Splitting the Dataset
X=df.iloc[:, :df.shape[1]-1]
Y=df.iloc[:, -1]
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.20, random_state=8) 

# //4
def apply_model(model):#Model to print the scores of various models
    model.fit(X_train,Y_train)
    print("Training score = ",model.score(X_train,Y_train))
    print("Testing score = ",model.score(X_test,Y_test))
    print("Accuracy = ",model.score(X_test,Y_test))
    Y_pred = model.predict(X_test)
    print("Predicted values:\n",Y_pred)
    print("Confusion Matrix:\n",confusion_matrix(Y_test,Y_pred))
    print("Classification Report:\n",classification_report(Y_test,Y_pred))
    
# //5
    knn = KNeighborsClassifier(n_neighbors=17) #KNN Model
    apply_model(knn)
# //6
svm = SVC(kernel='linear',random_state=3,max_iter=10000) #SVM Model
apply_model(svm)
    