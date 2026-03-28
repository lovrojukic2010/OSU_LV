import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

#a
data = pd.read_csv('data_C02_emission.csv')
y=data['CO2 Emissions (g/km)'].copy()
X=data[['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)']]
X_train , X_test , y_train , y_test = train_test_split (X, y, test_size = 0.2, random_state =1) #vraca isti tip kao i predani
#inace je lakse raditi s numpy arrayevima nego s dataframeom jer stalno sve vraca numpy

#b
for col in X_train.columns:
    plt.scatter(X_train[col],y_train, c='b', label='Train', s=5)
    plt.scatter(X_test[col],y_test, c='r', label='Test', s=5)
    plt.xlabel(col)
    plt.ylabel('CO2 Emissions (g/km)')
    plt.legend()
    plt.show()

#c
sc = MinMaxScaler () #transform vraca numpy array, zato se mora nazad u dataframe (da se sve radilo s numpy, ne bi bilo potrebe za time)
X_train_n = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
for col in X_train.columns:
    fig,axs = plt.subplots(2,figsize=(8, 8))
    axs[0].hist(X_train[col])
    axs[0].set_title('Before scaler')
    axs[1].hist(X_train_n[col])
    axs[1].set_xlabel(col)
    axs[1].set_title('After scaler')
    plt.show()
X_test_n = pd.DataFrame(sc.transform(X_test), columns=X_test.columns, index=X_test.index)

#d
linearModel=lm.LinearRegression()
linearModel.fit(X_train_n, y_train)
print(f'Parametri modela: {linearModel.coef_}')
print(f'Intercept parametar: {linearModel.intercept_}')

#e
y_prediction = linearModel.predict(X_test_n) #vraca numpy array
plt.scatter(X_test_n['Fuel Consumption City (L/100km)'],y_test, c='b', label='Real values', s=5)
plt.scatter(X_test_n['Fuel Consumption City (L/100km)'],y_prediction, c='r', label='Prediction', s=5)
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.legend()
plt.show()

#f
print(f'Mean squared error: {mean_squared_error(y_test, y_prediction)}')
print(f'Mean absolute error: {mean_absolute_error(y_test, y_prediction)}')
print(f'Mean absolute percentage error: {mean_absolute_percentage_error(y_test, y_prediction)}%')
print(f'R2 score: {r2_score(y_test, y_prediction)}')
