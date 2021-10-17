import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('indian_flatsell.csv')
x = df.head()

#print(x)
y = df.info()
#print(y)
z = df.describe()
#print(z)
a = df.columns
print(a)
sns.pairplot(df)
plt.show()

X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population', 'Price', 'Address']]
Y = df['Price']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.40,random_state=102)
#print(X_train)


lm = LinearRegression()
print(lm.fit(X_train, Y_train))
lm.fit(X_train, Y_train)

predictions = lm.predict(X_test)
plt.scatter(Y_test, predictions)
plt.show()
