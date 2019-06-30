import os
from CNN import CNN
from metrics import *

# only A and B
categories = ['A', 'B']
Img_Size = [100, 100]


def data_dirs(filename):
    """
    Citations: Python documentation was referred for understanding directory traversing using os
    :param filename: the current filename - in this case it point to this file -> signs1.py
    :return: current direcotory, train data and test data directory paths
    """
    current_dir = os.path.dirname(os.path.abspath(filename))
    os.chdir('../data')
    data_dir = os.getcwd()
    train_data_dir = data_dir + '/asl_alphabet_train/'
    test_data_dir = data_dir + '/asl_alphabet_test/'

    return current_dir, train_data_dir, test_data_dir


def main():
    """
    This is the wrapper function which calls and generates data from other classes and helper functions
    :return: does specifically return anything.
    """

    # get current and data directories (train and test)
    current_dir, train_data_dir, test_data_dir = data_dirs(__file__)

    # create an object of CNN and use it to create training and testing data along with their corresponding labels
    cnn = CNN(categories, train_data_dir, test_data_dir)
    train_matrix, train_labels = cnn.create_training_data()
    test_matrix, test_labels = cnn.create_testing_data()

    # resize images with the earlier defined size and pass in the matrices to be resized
    train_matrix = cnn.resize_images(Img_Size[0], Img_Size[1], train_matrix)
    test_matrix = cnn.resize_images(Img_Size[0], Img_Size[1], test_matrix)

    # convert data to numpy array
    train_data = cnn.create_numpy_data(train_matrix)
    test_data = cnn.create_numpy_data(test_matrix)

    # reshaping the images in order to make it ready for convolution
    input_shape, train_data, __, _ = cnn.data_reshape(train_data, train_labels, 1)
    ___, test_data, classes, nClasses = cnn.data_reshape(test_data, test_labels, 1)

    # convert into float32 type
    train_data = cnn.change_datatype(train_data, 'float32')
    test_data = cnn.change_datatype(test_data, 'float32')

    # scaling data
    train_data = cnn.scale_images(train_data, 255)
    test_data = cnn.scale_images(test_data, 255)

    # one hot encoding labels
    train_labels = cnn.do_one_hot_encoding(train_labels)
    test_labels = cnn.do_one_hot_encoding(test_labels)

    # create CNN model. we have 32 filters in each layer, with activation function as sigmoid, along with filter size as
    model = cnn.create_model(input_shape, activation="sigmoid", no_of_filters=32, filter_size=3, pool_size=2,
                             dense_layers=2)

    # compile model with setting these hyper parameters
    model = cnn.compile_model(model=model, loss="binary_crossentropy", optimizer="adam",
                              metrics=['accuracy', 'mse', 'mae', 'cosine', 'mape'])

    # fit model and generate history
    history = cnn.fit_model(model=model, train_data=train_data, train_labels=train_labels, batch_size=30, epochs=2,
                            verbose=1, splits=0.3)
    print("History", history)

    # find out model scores
    scores = cnn.evaluate_model(model=model, test_data=test_data, test_labels=test_labels)
    print("Scores", scores)

    # find testing accuracy and predicted values
    predicted_values = cnn.predict_using_model(model=model, test_data=test_data, test_labels=test_labels)
    print(predicted_values)


    ## Estimate the model performance using the following metrics

    mse = mean_squared_error(history, True)
    print(mse)

    mae = mean_absolute_error(history, True)
    print(mae)

    mape = mean_absolute_percentage_error(history, True)
    print(mape)

    train_acc, val_acc = train_validation_accuracy(history, True)

    train_loss, val_loss = train_validation_loss(history, True)
    print(train_acc, val_acc, train_loss, val_loss)

    confusion(test_labels.argmax(axis=1), predicted_values.round().argmax(axis=1), True)

    # lloss = l_loss(test_labels, predicted_values, True)
    # print(lloss)
    # print("Test",test_labels)
    # print("Predicted",predicted_values)
    # f1 = f1_score(test_labels, predicted_values)
    # print(f1)

    # r = auc_roc(test_labels, predicted_values, True)


if __name__ == '__main__':
    main()
