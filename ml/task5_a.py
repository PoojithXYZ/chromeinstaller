
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline


df = pd.read_csv('Salary_Data.csv')
df
df.head()

plt.scatter(df['YearsExperience'],df['Salary'])
plt.xlabel("YearsExperience")
plt.ylabel("Salary")

plt.show()

df.corr()

import seaborn as sns
sns.pairplot(df)
plt.show()

x = df[['YearsExperience']]
y = df['Salary']

x_series=df['YearsExperience']
np.array(x_series).shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
x_test

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

print("Coefficient or slope:",regressor.coef_)
print("Intercept:",regressor.intercept_)

plt.scatter(x_train,y_train)
plt.plot(x_train,regressor.predict(x_train))

plt.show()

y_pred = regressor.predict(x_test)
y_pred

from sklearn.metrics import mean_absolute_error
print("MAE",mean_absolute_error(y_test,y_pred))

from sklearn.metrics import mean_squared_error
print("MSE",mean_squared_error(y_test,y_pred))

print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred)))

from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print(score)

1 - (1-score)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)



