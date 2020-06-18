'''objective-2'''
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

#function to fill missing value using Imputer class
def miss(X):
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(X)
    X = imputer.transform(X)
    return X

#splitting of dataset
def split(X,Y):    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    return  X_train, X_test, y_train, y_test

#importing dataset
czech=pd.read_csv("czech.csv")
germany=pd.read_csv("germany.csv")
indonesia=pd.read_csv("indonesia.csv")
italy=pd.read_csv("italy.csv")
phillipines=pd.read_csv("phillipines.csv")
southkorea=pd.read_csv("southkorea.csv")
spain=pd.read_csv("spain.csv")
#checking null value
print("Checking the missing values of dataset:",czech.isnull())
print("Checking the missing values of dataset:",germany.isnull())
print("Checking the missing values of dataset:",indonesia.isnull())
print("Checking the missing values of dataset:",italy.isnull())
print("Checking the missing values of dataset:",phillipines.isnull())
print("Checking the missing values of dataset:",southkorea.isnull())
print("Checking the missing values of dataset:",spain.isnull())


"""Country-1 Italy"""

#reading data into arrays
X_days= italy.iloc[:,-1].values
y_total_cases= italy.iloc[:,1].values
y_deaths=italy.iloc[:,2].values
#filling the missing value
X_days=miss(X_days.reshape(-1,1))
y_total_cases= miss(y_total_cases.reshape(-1,1))
y_deaths=miss(y_deaths.reshape(-1,1))
#transforming normal data into polynomial data 
poly_reg = PolynomialFeatures(degree =4)
X_poly=poly_reg.fit_transform(X_days.reshape(-1,1))
#splitting dataset for prediction of total cases
X_train_totalcases, X_test_totalcases, y_train_totalcases, y_test_totalcases=split(X_poly,y_total_cases)
#splitting dataset for prediction of total  deaths cases
X_train_deaths, X_test_deaths, y_train_deaths, y_test_deaths=split(X_poly,y_deaths)
#fitting regressor 
Lin_reg_totalcases=LinearRegression()
Lin_reg_totalcases.fit(X_train_totalcases,y_train_totalcases)
#prediction of test set
y_pred_totalcases=Lin_reg_totalcases.predict(X_test_totalcases);
#prediction accuracy
from sklearn.metrics import r2_score
r2_score(y_test_totalcases,y_pred_totalcases)

#predicting future dataset
"""predicting data till 31st july i.e next 84 days"""
X_upcoming_days = []
for i in range(92):
    X_upcoming_days.append(i)
X_upcoming_days=np.array(X_upcoming_days).reshape(-1,1);

y_pred_future=Lin_reg_totalcases.predict(poly_reg.fit_transform(X_upcoming_days))


#visualizing future predictions
plt.figure(figsize=(15,5))
X_grid = np.arange(min(X_upcoming_days), max(X_upcoming_days), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_days,y_total_cases, color = 'red',label="Total Cases In Italy till 12th June 2020")
plt.plot(X_grid, Lin_reg_totalcases.predict(poly_reg.fit_transform(X_grid)), color = 'blue',label="Future Predicton")
plt.title('Predicton of Covid 19  cases in Italy  till 31st July 2020')
plt.xlabel('days')
plt.ylabel('cases')
plt.legend()
plt.show()


#fitting regressor
Lin_reg_deaths=LinearRegression()
Lin_reg_deaths.fit(X_train_deaths,y_train_deaths)

#predicting test set
y_pred_deaths=Lin_reg_deaths.predict(X_test_deaths);

#accuracy
r2_score(y_test_deaths,y_pred_deaths)
#predicting future dataset
y_pred_future_death=Lin_reg_deaths.predict(poly_reg.fit_transform(X_upcoming_days))

#visualizing future predictions
plt.figure(figsize=(15,5))
X_grid = np.arange(min(X_upcoming_days), max(X_upcoming_days), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_days,y_deaths, color = 'red',label="Death Cases in Italy ")
plt.plot(X_grid, Lin_reg_deaths.predict(poly_reg.fit_transform(X_grid)), color = 'blue',label="Predicting Death Cases In Italy")
plt.title('predicton of deaths due Covid 19  till 31st July 2020')
plt.xlabel('days')
plt.ylabel('deaths')
plt.legend()
plt.show()

#predicting for Age-Group 30-50
age=pd.read_csv('Age Group Covid Info.csv')
percentagedata=(age[age.Country=="Italy"]["%Age Share"].values)/100
totalcases_agegroup=(age[age.Country=="Italy"]["Confirmed Cases"].values)
predictions=y_pred_future_death[45:92]*percentagedata

#observations

#to generate date series
data=pd.date_range(start='15/06/2020' ,end='31/07/2020').date
#temporary data frame to hold all dates
df=pd.DataFrame(data,columns=["Observation_date"])
total_time_period=pd.to_datetime(df["Observation_date"]).dt.date

#Visualizing total death cases of Age group 30-50 till 31st July
plt.figure(figsize=(18,9))
sns.barplot(x=total_time_period,y=predictions.flatten())
plt.title("Total Covid-19 Cases(Age-Group 30-50) in Italy till 31st July ")
plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.xticks(rotation=90)

#DEATH RATE
deathrate=(predictions/totalcases_agegroup)*100
#Visualizing death rate of Age group 30-50 till 31st July
plt.figure(figsize=(15,5))
sns.lineplot(x=total_time_period,y=predictions.flatten())
plt.title("Total Covid-19 Cases(Age-Group 30-50) in Italy till 31st July ")
plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.xticks(rotation=90)

predictions_italy=deathrate

"""Country-2 Czech"""

X_days= czech.iloc[:,-1].values
y_total_cases= czech.iloc[:,1].values
y_deaths=czech.iloc[:,2].values



poly_reg = PolynomialFeatures(degree =4)
X_poly=poly_reg.fit_transform(X_days.reshape(-1,1))

X_train_totalcases, X_test_totalcases, y_train_totalcases, y_test_totalcases=split(X_poly,y_total_cases)

X_train_deaths, X_test_deaths, y_train_deaths, y_test_deaths=split(X_poly,y_deaths)

Lin_reg_totalcases=LinearRegression()
Lin_reg_totalcases.fit(X_train_totalcases,y_train_totalcases)

y_pred_totalcases=Lin_reg_totalcases.predict(X_test_totalcases);

from sklearn.metrics import r2_score
r2_score(y_test_totalcases,y_pred_totalcases)


y_pred_future=Lin_reg_totalcases.predict(poly_reg.fit_transform(X_upcoming_days))


#visualizing future predictions
plt.figure(figsize=(15,5))
X_grid = np.arange(min(X_upcoming_days), max(X_upcoming_days), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_days,y_total_cases, color = 'red',label="Total Cases In Czech till 12th June 2020")
plt.plot(X_grid, Lin_reg_totalcases.predict(poly_reg.fit_transform(X_grid)), color = 'blue',label="Future Predicton")
plt.title('Predicton of Covid 19  cases in Czech  till 31st July 2020')
plt.xlabel('days')
plt.ylabel('cases')
plt.legend()
plt.show()


#fitting regressor
Lin_reg_deaths=LinearRegression()
Lin_reg_deaths.fit(X_train_deaths,y_train_deaths)

#predicting test set
y_pred_deaths=Lin_reg_deaths.predict(X_test_deaths);

#accuracy
r2_score(y_test_deaths,y_pred_deaths)
#predicting future dataset
y_pred_future_death=Lin_reg_deaths.predict(poly_reg.fit_transform(X_upcoming_days))

#visualizing future predictions
plt.figure(figsize=(15,5))
X_grid = np.arange(min(X_upcoming_days), max(X_upcoming_days), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_days,y_deaths, color = 'red',label="Death Cases in Czech ")
plt.plot(X_grid, Lin_reg_deaths.predict(poly_reg.fit_transform(X_grid)), color = 'blue',label="Predicting Death Cases In Czech")
plt.title('predicton of deaths due Covid 19  till 31st July 2020')
plt.xlabel('days')
plt.ylabel('deaths')
plt.legend()
plt.show()

#predicting for Age-Group 30-50
age=pd.read_csv('Age Group Covid Info.csv')
percentagedata=(age[age.Country=="Czechia"]["%Age Share"].values)/100
totalcases_agegroup=(age[age.Country=="Czechia"]["Confirmed Cases"].values)
predictions=y_pred_future_death[45:92]*percentagedata


#Visualizing total death cases of Age group 30-50 till 31st July
plt.figure(figsize=(18,9))
sns.barplot(x=total_time_period,y=predictions.flatten())
plt.title("Total Covid-19 Cases(Age-Group 30-50) in Czech till 31st July ")
plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.xticks(rotation=90)

#DEATH RATE
deathrate=(predictions/totalcases_agegroup)*100
#Visualizing death rate of Age group 30-50 till 31st July
plt.figure(figsize=(15,5))
sns.lineplot(x=total_time_period,y=predictions.flatten())
plt.title("Total Covid-19 Cases(Age-Group 30-50) in Czech till 31st July ")
plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.xticks(rotation=90)

predictions_czech=deathrate


"""Country -3 Spain"""

X_days= spain.iloc[:,-1].values
y_total_cases= spain.iloc[:,1].values
y_deaths=spain.iloc[:,2].values



poly_reg = PolynomialFeatures(degree =3)
X_poly=poly_reg.fit_transform(X_days.reshape(-1,1))

X_train_totalcases, X_test_totalcases, y_train_totalcases, y_test_totalcases=split(X_poly,y_total_cases)

X_train_deaths, X_test_deaths, y_train_deaths, y_test_deaths=split(X_poly,y_deaths)

Lin_reg_totalcases=LinearRegression()
Lin_reg_totalcases.fit(X_train_totalcases,y_train_totalcases)

y_pred_totalcases=Lin_reg_totalcases.predict(X_test_totalcases);

from sklearn.metrics import r2_score
r2_score(y_test_totalcases,y_pred_totalcases)


y_pred_future=Lin_reg_totalcases.predict(poly_reg.fit_transform(X_upcoming_days))


#visualizing future predictions
plt.figure(figsize=(15,5))
X_grid = np.arange(min(X_upcoming_days), max(X_upcoming_days), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_days,y_total_cases, color = 'red',label="Total Cases In Spain till 12th June 2020")
plt.plot(X_grid, Lin_reg_totalcases.predict(poly_reg.fit_transform(X_grid)), color = 'blue',label="Future Predicton")
plt.title('Predicton of Covid 19  cases in Spain  till 31st July 2020')
plt.xlabel('days')
plt.ylabel('cases')
plt.legend()
plt.show()


Lin_reg_deaths=LinearRegression()
Lin_reg_deaths.fit(X_train_deaths,y_train_deaths)

y_pred_deaths=Lin_reg_deaths.predict(X_test_deaths);

r2_score(y_test_deaths,y_pred_deaths)
y_pred_future_death=Lin_reg_deaths.predict(poly_reg.fit_transform(X_upcoming_days))

plt.figure(figsize=(15,5))
X_grid = np.arange(min(X_upcoming_days), max(X_upcoming_days), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_days,y_deaths, color = 'red',label="Death Cases in Spain ")
plt.plot(X_grid, Lin_reg_deaths.predict(poly_reg.fit_transform(X_grid)), color = 'blue',label="Predicting Death Cases In Czech")
plt.title('predicton of deaths due Covid 19  till 31st July 2020')
plt.xlabel('days')
plt.ylabel('deaths')
plt.legend()
plt.show()

#predicting for Age-Group 30-50
age=pd.read_csv('Age Group Covid Info.csv')
percentagedata=(age[age.Country=="Spain"]["%Age Share"].values)/100
totalcases_agegroup=(age[age.Country=="Spain"]["Confirmed Cases"].values)
predictions=y_pred_future_death[45:92]*percentagedata


#Visualizing total death cases of Age group 30-50 till 31st July
plt.figure(figsize=(15,5))
sns.barplot(x=total_time_period,y=predictions.flatten())
plt.title("Total Covid-19 Cases(Age-Group 30-50) in Spain till 31st July ")
plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.xticks(rotation=90)

#DEATH RATE
deathrate=(predictions/totalcases_agegroup)*100
#Visualizing death rate of Age group 30-50 till 31st July
plt.figure(figsize=(15,5))
sns.lineplot(x=total_time_period,y=predictions.flatten())
plt.title("Total Covid-19 Cases(Age-Group 30-50) in Spain till 31st July ")
plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.xticks(rotation=90)

predictions_spain=deathrate

"""Country 4 Phillipines"""

X_days= phillipines.iloc[:,-1].values
y_total_cases= phillipines.iloc[:,1].values
y_deaths=phillipines.iloc[:,2].values



poly_reg = PolynomialFeatures(degree =4)
X_poly=poly_reg.fit_transform(X_days.reshape(-1,1))

X_train_totalcases, X_test_totalcases, y_train_totalcases, y_test_totalcases=split(X_poly,y_total_cases)

X_train_deaths, X_test_deaths, y_train_deaths, y_test_deaths=split(X_poly,y_deaths)

Lin_reg_totalcases=LinearRegression()
Lin_reg_totalcases.fit(X_train_totalcases,y_train_totalcases)

y_pred_totalcases=Lin_reg_totalcases.predict(X_test_totalcases);

from sklearn.metrics import r2_score
r2_score(y_test_totalcases,y_pred_totalcases)


y_pred_future=Lin_reg_totalcases.predict(poly_reg.fit_transform(X_upcoming_days))


#visualizing future predictions
plt.figure(figsize=(15,5))
X_grid = np.arange(min(X_upcoming_days), max(X_upcoming_days), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_days,y_total_cases, color = 'red',label="Total Cases In phillipines till 12th June 2020")
plt.plot(X_grid, Lin_reg_totalcases.predict(poly_reg.fit_transform(X_grid)), color = 'blue',label="Future Predicton")
plt.title('Predicton of Covid 19  cases in phillipines  till 31st July 2020')
plt.xlabel('days')
plt.ylabel('cases')
plt.legend()
plt.show()


Lin_reg_deaths=LinearRegression()
Lin_reg_deaths.fit(X_train_deaths,y_train_deaths)

y_pred_deaths=Lin_reg_deaths.predict(X_test_deaths);

r2_score(y_test_deaths,y_pred_deaths)
y_pred_future_death=Lin_reg_deaths.predict(poly_reg.fit_transform(X_upcoming_days))

plt.figure(figsize=(15,5))
X_grid = np.arange(min(X_upcoming_days), max(X_upcoming_days), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_days,y_deaths, color = 'red',label="Death Cases in Phillipines ")
plt.plot(X_grid, Lin_reg_deaths.predict(poly_reg.fit_transform(X_grid)), color = 'blue',label="Predicting Death Cases In Czech")
plt.title('predicton of deaths due Covid 19  till 31st July 2020')
plt.xlabel('days')
plt.ylabel('deaths')
plt.legend()
plt.show()

#predicting for Age-Group 30-50
age=pd.read_csv('Age Group Covid Info.csv')
percentagedata=(age[age.Country=="Phillipines"]["%Age Share"].values)/100
totalcases_agegroup=(age[age.Country=="Phillipines"]["Confirmed Cases"].values)
predictions=y_pred_future_death[45:92]*percentagedata


#Visualizing total death cases of Age group 30-50 till 31st July
plt.figure(figsize=(18,9))
sns.barplot(x=total_time_period,y=predictions.flatten())
plt.title("Total Covid-19 Cases(Age-Group 30-50) in Phillipines till 31st July ")
plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.xticks(rotation=90)

#DEATH RATE
deathrate=(predictions/totalcases_agegroup)*100
#Visualizing death rate of Age group 30-50 till 31st July
plt.figure(figsize=(15,5))
sns.lineplot(x=total_time_period,y=predictions.flatten())
plt.title("Total Covid-19 Cases(Age-Group 30-50) in Phillipines till 31st July ")
plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.xticks(rotation=90)
predictions_phillip=deathrate

"""Country 5 SouthKorea"""

X_days= southkorea.iloc[:,-1].values
y_total_cases= southkorea.iloc[:,1].values
y_deaths=southkorea.iloc[:,2].values



poly_reg = PolynomialFeatures(degree =3)
X_poly=poly_reg.fit_transform(X_days.reshape(-1,1))

X_train_totalcases, X_test_totalcases, y_train_totalcases, y_test_totalcases=split(X_poly,y_total_cases)

X_train_deaths, X_test_deaths, y_train_deaths, y_test_deaths=split(X_poly,y_deaths)

Lin_reg_totalcases=LinearRegression()
Lin_reg_totalcases.fit(X_train_totalcases,y_train_totalcases)

y_pred_totalcases=Lin_reg_totalcases.predict(X_test_totalcases);

from sklearn.metrics import r2_score
r2_score(y_test_totalcases,y_pred_totalcases)


y_pred_future=Lin_reg_totalcases.predict(poly_reg.fit_transform(X_upcoming_days))


#visualizing future predictions
plt.figure(figsize=(15,5))
X_grid = np.arange(min(X_upcoming_days), max(X_upcoming_days), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_days,y_total_cases, color = 'red',label="Total Cases In south korea till 12th June 2020")
plt.plot(X_grid, Lin_reg_totalcases.predict(poly_reg.fit_transform(X_grid)), color = 'blue',label="Future Predicton")
plt.title('Predicton of Covid 19  cases in south korea  till 31st July 2020')
plt.xlabel('days')
plt.ylabel('cases')
plt.legend()
plt.show()


Lin_reg_deaths=LinearRegression()
Lin_reg_deaths.fit(X_train_deaths,y_train_deaths)

y_pred_deaths=Lin_reg_deaths.predict(X_test_deaths);

r2_score(y_test_deaths,y_pred_deaths)
y_pred_future_death=Lin_reg_deaths.predict(poly_reg.fit_transform(X_upcoming_days))

plt.figure(figsize=(15,5))
X_grid = np.arange(min(X_upcoming_days), max(X_upcoming_days), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_days,y_deaths, color = 'red',label="Death Cases in south korea ")
plt.plot(X_grid, Lin_reg_deaths.predict(poly_reg.fit_transform(X_grid)), color = 'blue',label="Predicting Death Cases In South Korea")
plt.title('predicton of deaths due Covid 19  till 31st July 2020')
plt.xlabel('days')
plt.ylabel('deaths')
plt.legend()
plt.show()

#predicting for Age-Group 30-50
age=pd.read_csv('Age Group Covid Info.csv')
percentagedata=(age[age.Country=="South Korea"]["%Age Share"].values)/100
totalcases_agegroup=(age[age.Country=="South Korea"]["Confirmed Cases"].values)
predictions=y_pred_future_death[45:92]*percentagedata


#Visualizing total death cases of Age group 30-50 till 31st July
plt.figure(figsize=(18,9))
sns.barplot(x=total_time_period,y=predictions.flatten())
plt.title("Total Covid-19 Cases(Age-Group 30-50) in South Korea till 31st July ")
plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.xticks(rotation=90)

#DEATH RATE
deathrate=(predictions/totalcases_agegroup)*100
#Visualizing death rate of Age group 30-50 till 31st July
plt.figure(figsize=(15,5))
sns.lineplot(x=total_time_period,y=predictions.flatten())
plt.title("Total Covid-19 Cases(Age-Group 30-50) in South Korea till 31st July ")
plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.xticks(rotation=90)
predictions_korea=deathrate

"""Country 6 Indonesia"""

X_days= indonesia.iloc[:,-1].values
y_total_cases= indonesia.iloc[:,1].values
y_deaths=indonesia.iloc[:,2].values



poly_reg = PolynomialFeatures(degree =4)
X_poly=poly_reg.fit_transform(X_days.reshape(-1,1))

X_train_totalcases, X_test_totalcases, y_train_totalcases, y_test_totalcases=split(X_poly,y_total_cases)

X_train_deaths, X_test_deaths, y_train_deaths, y_test_deaths=split(X_poly,y_deaths)

Lin_reg_totalcases=LinearRegression()
Lin_reg_totalcases.fit(X_train_totalcases,y_train_totalcases)

y_pred_totalcases=Lin_reg_totalcases.predict(X_test_totalcases);

from sklearn.metrics import r2_score
r2_score(y_test_totalcases,y_pred_totalcases)


y_pred_future=Lin_reg_totalcases.predict(poly_reg.fit_transform(X_upcoming_days))


#visualizing future predictions
plt.figure(figsize=(15,5))
X_grid = np.arange(min(X_upcoming_days), max(X_upcoming_days), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_days,y_total_cases, color = 'red',label="Total Cases In Indonesia till 12th June 2020")
plt.plot(X_grid, Lin_reg_totalcases.predict(poly_reg.fit_transform(X_grid)), color = 'blue',label="Future Predicton")
plt.title('Predicton of Covid 19  cases in Indonesia  till 31st July 2020')
plt.xlabel('days')
plt.ylabel('cases')
plt.legend()
plt.show()


Lin_reg_deaths=LinearRegression()
Lin_reg_deaths.fit(X_train_deaths,y_train_deaths)

y_pred_deaths=Lin_reg_deaths.predict(X_test_deaths);

r2_score(y_test_deaths,y_pred_deaths)
y_pred_future_death=Lin_reg_deaths.predict(poly_reg.fit_transform(X_upcoming_days))

plt.figure(figsize=(15,5))
X_grid = np.arange(min(X_upcoming_days), max(X_upcoming_days), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_days,y_deaths, color = 'red',label="Death Cases in Indonesia ")
plt.plot(X_grid, Lin_reg_deaths.predict(poly_reg.fit_transform(X_grid)), color = 'blue',label="Predicting Death Cases In Indonesia")
plt.title('predicton of deaths due Covid 19  till 31st July 2020')
plt.xlabel('days')
plt.ylabel('deaths')
plt.legend()
plt.show()

#predicting for Age-Group 30-50
age=pd.read_csv('Age Group Covid Info.csv')
percentagedata=(age[age.Country=="Indonesia"]["%Age Share"].values)/100
totalcases_agegroup=(age[age.Country=="Indonesia"]["Confirmed Cases"].values)
predictions=y_pred_future_death[45:92]*percentagedata


#Visualizing total death cases of Age group 30-50 till 31st July
plt.figure(figsize=(18,9))
sns.barplot(x=total_time_period,y=predictions.flatten())
plt.title("Total Covid-19 Cases(Age-Group 30-50) in Indonesia till 31st July ")
plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.xticks(rotation=90)

#DEATH RATE
deathrate=(predictions/totalcases_agegroup)*100
#Visualizing death rate of Age group 30-50 till 31st July
plt.figure(figsize=(15,5))
sns.lineplot(x=total_time_period,y=predictions.flatten())
plt.title("Total Covid-19 Cases(Age-Group 30-50) in Indonesia till 31st July ")
plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.xticks(rotation=90)
predictions_indo=deathrate

"""Country 7 Germany"""

X_days= germany.iloc[:,-1].values
y_total_cases= germany.iloc[:,1].values
y_deaths=germany.iloc[:,2].values



poly_reg = PolynomialFeatures(degree =3)
X_poly=poly_reg.fit_transform(X_days.reshape(-1,1))

X_train_totalcases, X_test_totalcases, y_train_totalcases, y_test_totalcases=split(X_poly,y_total_cases)

X_train_deaths, X_test_deaths, y_train_deaths, y_test_deaths=split(X_poly,y_deaths)

Lin_reg_totalcases=LinearRegression()
Lin_reg_totalcases.fit(X_train_totalcases,y_train_totalcases)

y_pred_totalcases=Lin_reg_totalcases.predict(X_test_totalcases);

from sklearn.metrics import r2_score
r2_score(y_test_totalcases,y_pred_totalcases)


y_pred_future=Lin_reg_totalcases.predict(poly_reg.fit_transform(X_upcoming_days))


#visualizing future predictions
plt.figure(figsize=(15,5))
X_grid = np.arange(min(X_upcoming_days), max(X_upcoming_days), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_days,y_total_cases, color = 'red',label="Total Cases In Germany till 12th June 2020")
plt.plot(X_grid, Lin_reg_totalcases.predict(poly_reg.fit_transform(X_grid)), color = 'blue',label="Future Predicton")
plt.title('Predicton of Covid 19  cases in Germany  till 31st July 2020')
plt.xlabel('days')
plt.ylabel('cases')
plt.legend()
plt.show()


Lin_reg_deaths=LinearRegression()
Lin_reg_deaths.fit(X_train_deaths,y_train_deaths)

y_pred_deaths=Lin_reg_deaths.predict(X_test_deaths);

r2_score(y_test_deaths,y_pred_deaths)
y_pred_future_death=Lin_reg_deaths.predict(poly_reg.fit_transform(X_upcoming_days))

plt.figure(figsize=(15,5))
X_grid = np.arange(min(X_upcoming_days), max(X_upcoming_days), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_days,y_deaths, color = 'red',label="Death Cases in Germany ")
plt.plot(X_grid, Lin_reg_deaths.predict(poly_reg.fit_transform(X_grid)), color = 'blue',label="Predicting Death Cases In Indonesia")
plt.title('predicton of deaths due Covid 19  till 31st July 2020')
plt.xlabel('days')
plt.ylabel('deaths')
plt.legend()
plt.show()

#predicting for Age-Group 30-50
age=pd.read_csv('Age Group Covid Info.csv')
percentagedata=(age[age.Country=="Germany"]["%Age Share"].values)/100
totalcases_agegroup=(age[age.Country=="Germany"]["Confirmed Cases"].values)
predictions=y_pred_future_death[45:92]*percentagedata


#Visualizing total death cases of Age group 30-50 till 31st July
plt.figure(figsize=(18,9))
sns.barplot(x=total_time_period,y=predictions.flatten())
plt.title("Total Covid-19 Cases(Age-Group 30-50) in Germany till 31st July ")
plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.xticks(rotation=90)

#DEATH RATE
deathrate=(predictions/totalcases_agegroup)*100
#Visualizing death rate of Age group 30-50 till 31st July
plt.figure(figsize=(15,5))
sns.lineplot(x=total_time_period,y=predictions.flatten())
plt.title("Total Covid-19 Cases(Age-Group 30-50) in Germany till 31st July ")
plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.xticks(rotation=90)
predictions_germany=deathrate

sns.lineplot(x=total_time_period,y=predictions_czech.flatten(),label="Czechia")
sns.lineplot(x=total_time_period,y=predictions_spain.flatten(),label="Spain")
sns.lineplot(x=total_time_period,y=predictions_phillip.flatten(),label="Phllipines")
sns.lineplot(x=total_time_period,y=predictions_indo.flatten(),label="Indonesia")
sns.lineplot(x=total_time_period,y=predictions_germany.flatten(),label="germany")
plt.xticks(rotation=90)
plt.title("Death Rate of Covid-19 (Age-Group 30-50) Countrywise till 31st July ")
plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.xticks(rotation=90)
