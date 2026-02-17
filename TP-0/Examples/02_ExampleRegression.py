'''
Created on 24 Feb 2024

@author: deckyal
'''

from keras.models import Sequential
from keras.layers import Dense 
from keras.optimizers import Adam, SGD

import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../"))

df=pd.read_csv('./weight-height.csv')
X=df[['Height']].values
y_true=df[['Weight']].values

print(X.shape)
print(y_true.shape)
print(X,y_true)

##modelling 
model = Sequential()
model.add(Dense(10, input_shape = (1,)))
model.add(Dense(1, input_shape = (10,)))

model.summary()

model.compile(Adam(lr=0.09), "mean_squared_error")

model.fit(X,y_true, epochs=250, batch_size = 110)

y_pred= model.predict(X)

df.plot(kind='scatter',
       x='Height',
       y='Weight', title='Weight and Height in adults')
plt.plot(X, y_pred, color='red', linewidth=3)
plt.show()


w= model.get_weights()
print(w[0],w[1])