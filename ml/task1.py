

import pandas as pd
df=pd.read_csv("C:\\Users\\Admin\\Documents\\22241A6664 ml lab\\iris.data.csv",index_col=0)
df

df=pd.read_csv("C:\\Users\\Admin\\Documents\\22241A6664 ml lab\\iris.data.csv")
df
df.head()
df.head(10)
df.tail(10)
df.size

#  --  [ OR ] -- #

marks={"English":[67,89,90,55],
       "Maths":[55,67,45,56],
       "IP":[66,78,89,90],
       "Chemistry":[45,56,67,65],
       "Biology":[54,65,76,87]}
print(marks)

result=pd.DataFrame(marks,index=["Athang","Sujata","Sushil","Sumedh"])
print("OUTPUT")
print("*****************Marksheet*****************")
print(result)

result.to_csv("result.csv")
df=pd.read_csv("result.csv")
print(df)

