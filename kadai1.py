# -*- coding: utf-8 -*-
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import RMSprop
# Input data
X_train = np.array([[0.0, 0.0,-1.0],
                    [1.0, 0.0,-1.0],
                    [0.0, 1.0,-1.0],
                    [1.0, 1.0, -1.0]])
# Teacher for input
Y_train = np.array([0.0,
                    1.0,
                    1.0,
                    0.0])

def my_bias(shape, dtype=None):
    return np.array([-1])
# Build model
model = Sequential()
output_count_layer0 = 2
model.add(Dense(output_count_layer0, input_shape=(3,),
                activation='sigmoid'))  # Need to specify input shape for input layer
output_count_layer1 = 1
model.add(Dense(output_count_layer1,bias=True,bias_initializer=my_bias,activation='linear'))
model.compile(loss='mean_squared_error',
              optimizer=RMSprop(), metrics=['accuracy'])

# Start training
BATCH_SIZE = 4
ITERATION = 5000
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE,
                    nb_epoch=ITERATION, verbose=0)

# Evaluate model
X_test = X_train
Y_test = Y_train
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
model.summary()
#重み出力
print(model.get_weights())

