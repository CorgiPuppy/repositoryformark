import tensorflow as tf
import keras
import numpy as np


model = keras.Sequential([
    keras.layers.Dense(units=4, input_shape=[2], activation='relu'),
    keras.layers.Dense(units=1, activation='relu')
])
model.compile(optimizer='adam', loss='mean_squared_error')

xs = np.array([[0,0],[0,1],[1,0],[1,1]])
ys = np.array([0,1,1,0])

model.fit(xs, ys, epochs=1000)

print(model.evaluate(xs, ys, verbose=2))
print(model.predict(xs))
