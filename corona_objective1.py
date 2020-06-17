
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns
import datetime as dt
from sklearn.preprocessing import Imputer

#importing dataset
dataset = pd.read_csv('Covid19Cases.csv')
dataset.head()
#checking null value
print("Checking the missing values of dataset:",dataset.isnull())

#reading data into arrays
X_days= dataset.iloc[:,0].values
y_total_cases= dataset.iloc[:,2].values
y_recovered=dataset.iloc[:,3].values
y_deaths=dataset.iloc[:,4].values

#function to fill missing value using Imputer class
def miss(X):
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(X)
    X = imputer.transform(X)
    return X

#filling the missing value
X_days=miss(X_days.reshape(-1,1))
y_total_cases= miss(y_total_cases.reshape(-1,1))
y_recovered=miss(y_recovered.reshape(-1,1))
y_deaths=miss(y_deaths.reshape(-1,1))
    
#transforming normal data into polynomial data 
poly_reg = PolynomialFeatures(degree =4)
X_poly=poly_reg.fit_transform(X_days.reshape(-1,1))


#splitting of dataset
def split(X,Y):    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    return  X_train, X_test, y_train, y_test

#splitting dataset for prediction of total cases
X_train_totalcases, X_test_totalcases, y_train_totalcases, y_test_totalcases=split(X_poly,y_total_cases)
#splitting dataset for prediction of total recovered cases
X_train_recovered, X_test_recovered, y_train_recovered, y_test_recovered=split(X_poly,y_recovered)

''' Determining Infection cases'''
#fitting regressor 
Lin_reg_totalcases=LinearRegression()
Lin_reg_totalcases.fit(X_train_totalcases,y_train_totalcases)

#prediction of test set
y_pred_totalcases=Lin_reg_totalcases.predict(X_test_totalcases);

#prediction accuracy
from sklearn.metrics import r2_score
r2_score(y_test_totalcases,y_pred_totalcases)


"""predicting data till 31st july i.e next 184 days"""
X_upcoming_days = []
for i in range(185):
    X_upcoming_days.append(i)
X_upcoming_days=np.array(X_upcoming_days).reshape(-1,1);

y_pred_future=Lin_reg_totalcases.predict(poly_reg.fit_transform(X_upcoming_days))


''' Determining Recovered Cases'''
#fitting regressor
Lin_reg_recovered=LinearRegression()
Lin_reg_recovered.fit(X_train_recovered,y_train_recovered)

#predicting test set
y_pred_recovered=Lin_reg_recovered.predict(X_test_recovered)
#accuracy
r2_score(y_test_recovered,y_pred_recovered)

#predicting future dataset

y_pred_future_recovered=Lin_reg_recovered.predict(poly_reg.fit_transform(X_upcoming_days))



''' Finding Infection and Recovery Rates '''
population=1379427000
recovery_rate_new=(y_pred_future_recovered/y_pred_future)*100
recovery_rate_original= (y_recovered/y_total_cases)*100
infection_rate_original=(y_total_cases/population)*100
infection_rate_new=(y_pred_future/population)*100

#Recovery Rate till 29th May
plt.plot(X_days,recovery_rate_original, color = 'red',label="Recovery rate in India ")
plt.title('Recovery Rate  till 29th May 2020')
plt.xlabel('days')
plt.ylabel('Recovery Rate')
plt.legend()
plt.show()
#Recovery Rate from 15th June to 31st July
plt.plot(X_upcoming_days[139:185,:],recovery_rate_new[139:185,:], color = 'red',label="Prediction of Recovery rate in India ")
plt.title('Prediction of Recovery Rate from 15th June to 31st July')
plt.xlabel('days')
plt.ylabel('Recovery Rate')
plt.legend()
plt.show()


#Infection Rate till 29th May
plt.plot(X_days,infection_rate_original, color = 'red',label="Infection Rate  ")
plt.title('Infection Rate  till 29th May 2020')
plt.xlabel('days')
plt.ylabel('Infection Rate')
plt.legend()
plt.show()

#Death Rate from 15th June till 31st July
plt.plot(X_upcoming_days[139:185,:],infection_rate_new[139:185,:], color = 'red',label="Prediction of Infection Rate ")
plt.title('Predition of Infection Rate from 15th June   till 31st July 2020')
plt.xlabel('days')
plt.ylabel('Infection Rate')
plt.legend()
plt.show()

'''
#observations
#Datewise Barplots

#to import date series from dataset
observation_date=pd.to_datetime(dataset["Date"]).dt.date
observe_cases=dataset.iloc[:,2].values

#to generate date series
data=pd.date_range(start='30/01/2020' ,end='31/07/2020').date
#temporary data frame to hold all dates
df=pd.DataFrame(data,columns=["Observation_date"])
total_time_period=pd.to_datetime(df["Observation_date"]).dt.date

#Total Covid-19 Cases in India till 29th May
plt.figure(figsize=(18,9))
sns.barplot(x=observation_date,y=observe_cases)
plt.title("Total Covid-19 Cases in India till 29th May")
plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.xticks(rotation=90)

#Total Covid-19 Cases in India till 31st July 2020
plt.figure(figsize=(15,5))
sns.barplot(x=total_time_period,y=y_pred_future.flatten())
plt.title("Total Covid-19 Cases in India till 31st July 2020")
plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.xticks(rotation=90)

#Total Covid-19 Recovery Rate in India(15th June to 31st July)
plt.figure(figsize=(15,5))
plt.plot(total_time_period[137:184],recovery_rate_new[137:184].flatten())
plt.title("Total Covid-19 Recovery Rate in India from 15th June till 31st July")
plt.xlabel("Date")
plt.ylabel("Recovery Rate")
plt.xticks(rotation=90)

#Total Covid-19 Infection  Rate in India(15th June to 31st July)
plt.figure(figsize=(15,5))
plt.plot(total_time_period[137:184],infection_rate_new[137:184].flatten())
plt.title("Total Covid-19 Infection Rate in India from 15th June till  31st July")
plt.xlabel("Date")
plt.ylabel("Infection Rate")
plt.xticks(rotation=90)

#prediction of 30th(122nd day) and 31st may(123rd day)
ans1=Lin_reg_totalcases.predict(poly_reg.fit_transform([[122]]))
ans2=Lin_reg_recovered.predict(poly_reg.fit_transform([[122]]))

print("Prediction of 30th May(future prediction) for total cases :",ans1)
print("Prediction of 30th May(future prediction) for total recovered cases :",ans2)

ans3=Lin_reg_totalcases.predict(poly_reg.fit_transform([[123]]))
ans4=Lin_reg_recovered.predict(poly_reg.fit_transform([[123]]))


print("Prediction of 31st May(future prediction) for total cases :",ans3)
print("Prediction of 31st May(future prediction) for total recovered cases :",ans4)

'''