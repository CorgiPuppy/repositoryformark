import tensorflow as tf
import keras as keras
import pandas as pd


data = pd.read_csv("./train.csv")
print(data.describe())

mnist = keras.datasets.mnist 

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

predictions = model(x_train[:1]).numpy()
print(predictions)

print(tf.nn.softmax(predictions).numpy())

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5)
print(history)

model.evaluate(x_test,  y_test, verbose=2)
