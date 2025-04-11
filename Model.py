import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


#reading the file using pandas
dataset=pd.read_csv("Crop_recommendation.csv")
data=pd.DataFrame(dataset)  #<-converting the data into the dataframe
scaler=StandardScaler()

label=LabelEncoder()  #<-label encoding
data["label"]=label.fit_transform(data["label"])

from sklearn.preprocessing import LabelEncoder

#spliting the data into x and y 
x=data.iloc[:,:-1]
y=data['label']
#split the data into the trainnig and testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)

scaler=StandardScaler()
scaled_x_train=scaler.fit_transform(x_train)
scaled_x_test=scaler.fit_transform(x_test)

#choosing the model
model=LogisticRegression()
model.fit(scaled_x_train,y_train) #<- traning the model

#Plotting the graph by using the matplotlib
# plt.scatter(x_train["humidity"], y_train, c=y_train, cmap='viridis')
# plt.xlabel("Humidity")
# plt.ylabel("Crop")
# plt.title("Humidity vs Crop (Colored by Crop Type)")
# plt.colorbar(label="Crop Class")
# plt.show()

#testing the model
prediction=model.predict(scaled_x_test)
#print(prediction)
#accuracy
# acc=accuracy_score(y_test,prediction)*100 
# print(acc,"%")   #here we can obtain the accuracy around 97.99

N=float(input("Enter The Nitrogen in ppm:"))
P=float(input("Enter The Phosphorus Amount in ppm:"))
K=float(input("Enter The Potassium Amount in ppm:"))
temperature=float(input("Enter The temprature in Celsius:"))
humidity=float(input("Enter The  value of Humidity:"))
ph=float(input("Enter The PH value:"))
rainfall=float(input("Enter The rainfall:"))

new_data=[[N,P,K,temperature,humidity,ph,rainfall]]
new_df = pd.DataFrame(new_data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
new_scaled = scaler.transform(new_df)
newpredict=model.predict(new_scaled)


#print(newpredict)
crop_name = label.inverse_transform(newpredict)
print("Recommended crop:", crop_name[0]) 
