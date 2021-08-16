import random
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense
from keras.optimizers import Adam
class DQN:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = 0.001
        
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.model = self.create_model()

#    def call(self):
#       return self.model

    def create_model(self):
        model = keras.Sequential()
        
        model.add(Dense(128, input_dim=self.input_size, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(self.output_size))
        
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        
        return model
    
    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.random() < self.epsilon:
            return random.randrange(0, self.output_size)
        return np.argmax(self.model.predict(state)[0])
