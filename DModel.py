from matplotlib import image
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.utils import class_weight
class DModel:
    def __init__(self, image_shape, n_classes):
        self.image_shape = image_shape
        self.n_classes = n_classes
        self.model = self.define_model()

    def define_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.image_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.n_classes, activation='softmax'))
        # compile model
        # opt = SGD(lr=0.01, momentum=0.9)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def fit(self, X, Y, verbosity=0):
        self.model.fit(X, Y,epochs=30, verbose=verbosity)
       
    def get_uncertainty(self, sampleX, sampleY):
        shape_format = (len(sampleX), sampleX[0].shape[0], sampleX[0].shape[1], sampleX[0].shape[2])
        sampleX = np.array(sampleX)
        sampleX = sampleX.reshape(shape_format)
        predictions = self.model.predict(sampleX)
        sum_uncertainty = 0
        for i in range(len(sampleX)):
            sum_uncertainty += 1.0 - predictions[i,sampleY[i]]
        return sum_uncertainty / len(sampleX)

    def save(self, path):
        self.model.save(path)
    
    def load(self, path):
        self.model = tf.keras.models.load_model(path)