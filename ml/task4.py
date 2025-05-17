
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("https://raw.githubusercontent.com/gchoi/Dataset/refs/heads/master/ToyotaCorolla.csv")
df

plt.scatter(df["Age"],df["Price"],c="red")
plt.title='scatter plot of price vs age of the cars'
plt.xlabel('Age(months)')
plt.ylabel('Price(Euros)')
plt.show()

plt.hist(df['KM'],color='red',edgecolor='white',bins=5)
plt.title='Histogram of KM'
plt.xlabel('KM')
plt.ylabel('frequency')

counts = [979, 120, 12]
fuelType = ('Petrol', 'Diesel', 'CNG')
index = np.arange(len(fuelType))
plt.bar(index,counts,color=['red','blue','cyan'])
plt.title="Bar plot of fuel types"
plt.xlabel('Fuel Types')
plt.ylabel('Frequency')
plt.xticks(index, fuelType, rotation = 90)
plt.show()

plt.scatter(df['Age'], df['Price'], c=df['MetColor'], s=df['MetColor'])

plt.title="Scatter Plot"

plt.xlabel('Age')
plt.ylabel('Price')
plt.colorbar()
plt.show()

plt.plot(df['Age'])
plt.plot(df['Price'])

plt.title="Scatter Plot"

plt.xlabel('Age')
plt.ylabel('price')
plt.show()