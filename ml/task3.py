
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
#%matplotlib inline

cancer = load_breast_cancer()
print(cancer.keys())
print(cancer['DESCR'])

df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
df.head()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
Scaled_data = scaler.transform(df)
Scaled_data

pca = PCA(n_components=2)
pca.fit(Scaled_data)
x_pca = pca.transform(Scaled_data)
print(Scaled_data.shape)
print(x_pca.shape)

print(Scaled_data,
x_pca, sep="\n")

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'])
plt.xlabel('First pca')
plt.ylabel('Second pca')

