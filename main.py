import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
boston = load_boston()
X=np.array(boston.data)
Y=np.array(boston.target)
print(X.shape,Y.shape)
x1,y1,x2,y2=train_test_split(X,Y,test_size=0.3,random_state=27)
model=tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=26,input_shape=[13]))
model.add(tf.keras.layers.Dense(units=13,activation='relu'))
model.add(tf.keras.layers.Dense(units=6,activation='relu'))
model.add(tf.keras.layers.Dense(units=3,activation='relu'))
model.add(tf.keras.layers.Dense(units=1,activation='relu'))
model.compile(optimizer='adam',
              loss='mse',
              metrics='mae')
model.fit(x1,x2,epochs=50,validation_split=0.2)
model.evaluate(y1,y2)
model.save("/content/drive/MyDrive/Boston"+"Mod.h5")