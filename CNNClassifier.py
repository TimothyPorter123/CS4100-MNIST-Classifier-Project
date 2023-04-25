import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


# Credit for CNN code template to https://data-flair.training/blogs/python-deep-learning-project-handwritten-digit-recognition/
class CNN_Classifier(object):
    def __init__(self):
        self.batch_size = 128
        self.num_classes = 10
        self.epochs = 10
        self.input_shape = (28, 28, 1)
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=self.input_shape))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        self.model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

    def train(self, trainingData, validationData):
        x_train = trainingData[0].reshape(trainingData[0].shape[0], 28, 28, 1)
        x_test = validationData[0].reshape(trainingData[0].shape[0], 28, 28, 1)

        y_train = keras.utils.to_categorical(trainingData[1], 10)
        y_test = keras.utils.to_categorical(validationData[1], self.num_classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        hist = self.model.fit(x_train, y_train,batch_size=self.batch_size,epochs=self.epochs,verbose=1,validation_data=(x_test, y_test))
        return self

    def classify(self, img):
        img = img.reshape(1,28,28,1)
        img = img/255.0
        return self.model.predict([img])[0]