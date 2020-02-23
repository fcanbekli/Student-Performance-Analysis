import pandas as pd
import tensorflow as tf
df = pd.read_csv('./data/student-por.csv', delimiter=";")
df['AVG_G'] = round((df['G1'] + df['G2'] + df['G3']) / 3)
X_train, x_test, Y_train, y_test = df.iloc[:500, :30], df.iloc[500:, :30], df.iloc[:500, 33], df.iloc[500:, 33]


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(30, input_shape=(30,)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1))

# TODO Read compile parameters from doc
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])

