import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import max_error
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('data_C02_emission.csv')
ohe=OneHotEncoder()
fuelTypeEncoded=ohe.fit_transform(data[['Fuel Type']]).toarray() #OneHotEncoder.fit_transform ocekuje 2d array(dataframe[[stupac(i)]]), ne moze 1d (series[stupac])
data[ohe.categories_[0]]=fuelTypeEncoded #kategorije su imena stupaca, pohranjene u listu lista kategorija atributa categories_, zato [0], tj. jedina lista kategorija iz prosle linije dobivena
output_variable='CO2 Emissions (g/km)'
y=data[output_variable].copy()
X = data.drop('CO2 Emissions (g/km)', axis=1)
X_train_all , X_test_all , y_train , y_test = train_test_split (X, y, test_size = 0.2, random_state =1)
#potrebne sve velicine iz dataframe za laksi pronalazak modela kasnije

#izdvajanje numerickih velicina
input_variables=['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)','D', 'E', 'X', 'Z']
X_train = X_train_all[input_variables]
X_test = X_test_all[input_variables]

linearModel = lm.LinearRegression()
linearModel.fit(X_train,y_train)
y_prediction = linearModel.predict(X_test)
plt.scatter(X_test['Fuel Consumption City (L/100km)'],y_test, c='b',label='Real values', s=5)
plt.scatter(X_test['Fuel Consumption City (L/100km)'],y_prediction, c='r',label='Prediction', s=5)
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.legend()
plt.show()

maxError = max_error(y_test,y_prediction)
print(f"Model vozila s najvecom greskom u predvidanju: {X_test_all[abs(y_test-y_prediction)==maxError]['Model'].iloc[0]}")
