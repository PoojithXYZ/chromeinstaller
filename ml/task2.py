
from sklearn.datasets import load_iris
df = load_iris()

import numpy as np
import pandas as pd
df = pd.DataFrame(data=df.data, columns=df.feature_names)
df

x = df.iloc[:,:4].values
y = df.iloc[:,-1].values
x
y

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
imputer.fit(x[:,:4])
x[:,:4] = imputer.transform(x[:,:4])
x

from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)
y

from sklearn.preprocessing import OneHotEncoder
y = OneHotEncoder().fit_transform(y.reshape(-1,1))
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

print(
x_train,
x_test,
y_train,
y_test,
sep="\n----------\n")

from sklearn.preprocessing import MinMaxScaler
x_train = MinMaxScaler().fit_transform(x_train)
x_test = MinMaxScaler().fit_transform(x_test)
x_train
x_test

