import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestClassifier
import pickle


df=pd.read_csv(r'prsk_jobs.csv')

print(df)

le = LabelEncoder()
 

df.Job= le.fit_transform(df.Job)
df.Employer = le.fit_transform(df.Employer)
df.Location = le.fit_transform(df.Location)
df.Salary = le.fit_transform(df.Salary)

df=df.rename(columns={'Available_for_UKR':'target'})


hw_scaled = minmax_scale(df[['Job','Employer','Location','Salary']], feature_range=(0,1))
 
df['Job']=hw_scaled[:,0]
df['Employer']=hw_scaled[:,1]
df['Location']=hw_scaled[:,2]
df['Salary']=hw_scaled[:,3]

x=df.drop('target',axis=1)
y=df['target']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.27,random_state=42,shuffle=True)

rforest=RandomForestClassifier()
rforest.fit(x_train,y_train)

#y_pred=rforest.predict(x_test)


#train_predict=rforest.predict(x_train)
#accuracy=accuracy_score(y_train,train_predict)
#print("Accuracy : ",accuracy*100,'%')

#test_predict=rforest.predict(x_test)
#accuracy=accuracy_score(y_test,test_predict)
#print("Accuracy : ",accuracy*100,'%')

# Make pickle file of our model
pickle.dump(rforest, open("model.pkl", "wb"))