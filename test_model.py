from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_shape=(5,), activation='relu'))
print("Model created successfully!")
