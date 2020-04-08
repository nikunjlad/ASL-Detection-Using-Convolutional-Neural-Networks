import os, cv2
import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.utils import to_categorical
from keras.models import Sequential


class CNN:

    def __init__(self, categories, train_data_dir, test_data_dir):
        """

        :param categories:
        :param train_data_dir:
        :param test_data_dir:
        """
        self.categories = categories
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.train_matrix = list()
        self.train_labels = list()
        self.test_matrix = list()
        self.test_labels = list()

    def create_training_data(self):

        for index1, Category in enumerate(self.categories):
            path = os.path.join(self.train_data_dir, Category)  # path to cats or dogs Category

            for img in os.listdir(path):
                try:
                    # reading the image in gray scale mode and resizing it
                    image = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

                    self.train_matrix.append(image)
                    self.train_labels.append(index1)
                except Exception as e:
                    pass

        return self.train_matrix, self.train_labels

    def create_testing_data(self):


        for index2, Category in enumerate(self.categories):
            path = os.path.join(self.test_data_dir, Category)  # path to cats or dogs Category

            for img in os.listdir(path):
                try:
                    image = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

                    self.test_matrix.append(image)
                    self.test_labels.append(index2)
                except Exception as e:
                    pass

        return self.test_matrix, self.test_labels

    def resize_images(self, x_size, y_size, image):

        if isinstance(image, list):
            image_matrix = list()
            for img in image:
                resized_image = cv2.resize(img, (x_size, y_size))
                image_matrix.append(resized_image)
            return image_matrix
        else:
            resized_image = cv2.resize(image, (x_size, y_size))
            return resized_image

    def create_numpy_data(self, data):

        numpy_data = np.array(data)
        return numpy_data

    def do_one_hot_encoding(self, just_labels):
        one_hot_encoded_labels = to_categorical(just_labels)

        return one_hot_encoded_labels

    def change_datatype(self, data, dataType):

        new_data = data.astype(dataType)
        return new_data

    def scale_images(self, data, factor):

        data /= factor
        return data

    def data_reshape(self, data, labels, image_depth):

        images = data

        # Find the unique numbers from the train labels
        classes = np.unique(labels)
        nClasses = len(classes)
        print('Total number of outputs : ', nClasses)
        print('Output classes : ', classes)

        nRows, nCols = images.shape[1:]
        print(nRows, nCols, images.shape[0])
        images = images.reshape(images.shape[0], nRows, nCols, image_depth)
        input_shape = (nRows, nCols, image_depth)

        return input_shape, images, classes, nClasses

    def create_model(self, input_shape, activation, no_of_filters, filter_size, pool_size, dense_layers):

        model = Sequential()

        model.add(Conv2D(no_of_filters, (filter_size, filter_size), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

        model.add(Conv2D(no_of_filters, (filter_size, filter_size)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

        model.add(Dense(dense_layers))
        model.add(Activation(activation))

        return model

    def compile_model(self, model, loss, optimizer, metrics):
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        return model

    def fit_model(self, model, train_data, train_labels, batch_size, epochs, verbose, splits):
        history = model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, verbose=verbose,
                            validation_split=splits)

        return history

    def evaluate_model(self, model, test_data, test_labels):
        scores = model.evaluate(test_data, test_labels)

        return scores

    def predict_using_model(self, model, test_data, test_labels):

        ypred = model.predict(x=test_data, verbose=1)

        return ypred

