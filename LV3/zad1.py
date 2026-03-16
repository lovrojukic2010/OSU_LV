import pandas as pd
data = pd.read_csv('data_C02_emission.csv')

def zad1a():
    print(f'Broj mjerenja: {len(data)}')
    print('Tipovi velicina:')
    print(data.dtypes)
    print(f'Broj dupliciranih vrijednosti:{data.duplicated().sum()}')
    print('Broj izostalih vrijednosti po stupcima:')
    print(data.isnull().sum())
    data['Make']=pd.Categorical(data['Make'])
    data['Vehicle Class']=pd.Categorical(data['Vehicle Class'])
    data['Transmission']=pd.Categorical(data['Transmission'])
    data['Fuel Type']=pd.Categorical(data['Fuel Type'])
    print(data.dtypes)


def zad1b():
    topAndBottom3=data.sort_values(by='Fuel Consumption City (L/100km)')
    print('Najmanja 3:')
    print(topAndBottom3[['Make','Model','Fuel Consumption City (L/100km)']].head(3))
    print('Najveća 3:')
    print(topAndBottom3[['Make','Model','Fuel Consumption City (L/100km)']].tail(3))

def zad1c():
    print(data[(data['Engine Size (L)']>=2.5) & (data['Engine Size (L)']<=3.5)].agg({'Model':'count', 'CO2 Emissions (g/km)':'mean'}))

def zad1d():
    print(f"Broj Audija:{len(data[data['Make']=='Audi'])}")
    print(f"Prosjecna emisija CO2 audija s 4 cilindra: {(data[(data['Make']=='Audi') & (data['Cylinders']==4)])['CO2 Emissions (g/km)'].mean()} (g/km)")

def zad1e():
    print(data.groupby('Cylinders').agg({'Make':'count', 'CO2 Emissions (g/km)':'mean'}).rename(columns={'Make': 'Car count'}))

def zad1f():
    print('Mean i median za dizel:')
    print(data[data['Fuel Type']=='D']['Fuel Consumption City (L/100km)'].agg(['mean','median']))
    print('Mean i median za benzin:')
    print(data[data['Fuel Type']=='X']['Fuel Consumption City (L/100km)'].agg(['mean','median']))

def zad1g():
    print('Vozilo s 4 cilindra dizel s najvećom potrošnjom u gradu:')
    print(data[(data['Cylinders']==4) & (data['Fuel Type']=='D')].sort_values(by='Fuel Consumption City (L/100km)').tail(1))

def zad1h():
    print(f"Broj vozila s manualnim mjenjačem: {len(data[data['Transmission'].str.startswith('M')])}")

def zad1i():
    print(data.corr(numeric_only=True))

#zad1a()
#zad1b()
#zad1c()
#zad1d()
#zad1e()
#zad1f()
#zad1g()
#zad1h()
#zad1i()

