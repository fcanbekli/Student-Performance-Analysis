import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#Import data from csv file
df = pd.read_csv('./data/student-por.csv', delimiter=";")


#Data Preprocessing
df['AVG_G'] = round((df['G1'] + df['G2'] + df['G3']) / 3)

labelEncoder = LabelEncoder()
df.iloc[:,0] = labelEncoder.fit_transform(df.iloc[:,0])
df.iloc[:,1] = labelEncoder.fit_transform(df.iloc[:,1])
df.iloc[:,3] = labelEncoder.fit_transform(df.iloc[:,3])
df.iloc[:,4] = labelEncoder.fit_transform(df.iloc[:,4])
df.iloc[:,5] = labelEncoder.fit_transform(df.iloc[:,5])
df.iloc[:,8] = labelEncoder.fit_transform(df.iloc[:,8])
df.iloc[:,9] = labelEncoder.fit_transform(df.iloc[:,9])
df.iloc[:,10] = labelEncoder.fit_transform(df.iloc[:,10])
df.iloc[:,11] = labelEncoder.fit_transform(df.iloc[:,11])
df.iloc[:,15] = labelEncoder.fit_transform(df.iloc[:,15])
df.iloc[:,16] = labelEncoder.fit_transform(df.iloc[:,15])
df.iloc[:,17] = labelEncoder.fit_transform(df.iloc[:,15])
df.iloc[:,18] = labelEncoder.fit_transform(df.iloc[:,15])
df.iloc[:,19] = labelEncoder.fit_transform(df.iloc[:,15])
df.iloc[:,20] = labelEncoder.fit_transform(df.iloc[:,15])
df.iloc[:,21] = labelEncoder.fit_transform(df.iloc[:,15])
df.iloc[:,22] = labelEncoder.fit_transform(df.iloc[:,15])


# Train and test set generation
xTrain, xTest, yTrain, yTest = train_test_split(df.iloc[:, :30], df.iloc[:, 33], test_size = 0.2, random_state = 0)
print(xTrain)

# Deep learning model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(30, input_shape=(30,)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1))

# Compilation model
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])
model.fit(xTrain, yTrain, epochs=100)
print("Evaluate")
print(model.evaluate(xTest,  yTest, verbose=2))
print("Prediction of the sample 100 from test set")
print(model.predict(xTest.iloc[:])[22])
print("yTest")
print(yTest.iloc[22])
model.save('model.h5')
