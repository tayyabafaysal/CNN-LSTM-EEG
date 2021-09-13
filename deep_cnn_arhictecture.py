from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten


# build the deep network architecture
def build_deep_cnn(input_shape):
    model = Sequential()

    model.add(Dense(232, input_dim=input_shape, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model