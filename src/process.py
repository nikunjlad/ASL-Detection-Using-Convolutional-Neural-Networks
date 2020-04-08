from six.moves import cPickle as pickle
import os, sys, platform, h5py
import numpy as np
import cv2
import random


class ProcessData:

    def __init__(self, data_path=None, categories=None):
        self.data_path = data_path
        self.categories = categories

    @staticmethod
    def load_pickle(f):
        """
        Read pickle file based on version of python
        :param f: pickle file to load
        :return: unpickled file
        """
        version = platform.python_version_tuple()  # get the version of the current python interpreter
        if version[0] == '2':
            # pickle command to load file if python version is 2
            return pickle.load(f)
        elif version[0] == '3':
            # pickle command to load file if python version is 3, use encoding latin1
            return pickle.load(f,
                               encoding='latin1')
        raise ValueError("invalid python version: {}".format(version))

    def load_CIFAR_batch(self, filename):
        """
        Load single batch of cifar. There are a total of 6 batches in form of pickle files
        :param filename: entire path to pickle filename
        :return: data X and corresponding labels y
        """
        print("Loading CIFAR batch...")
        with open(filename, 'rb') as f:
            datadict = self.load_pickle(f)  # function to unpickle a pickle file, which returns a dictionary
            X = datadict['data']  # data
            Y = datadict['labels']  # labels

            # reshape the X numpy array to a 4D array where each image is 32x32 with 3 channels and a total of 10000 images
            X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")  # realign column arguments
            Y = np.array(Y)  # convert labels from list to numpy array so as to be consistent with the datatype
            return X, Y

    def load_CIFAR10(self):
        """
        Load all the CIFAR10 dataset batches
        :param ROOT:  path to all the data batch files
        :return: training data and labels, testing data and labels
        """
        xs = []
        ys = []
        for b in range(1, 6):
            f = os.path.join(self.data_path, 'data_batch_' + str(b))  # full path to pickle data files
            X, Y = self.load_CIFAR_batch(f)  # load data batches
            xs.append(X)  # list of unpickled training data files
            ys.append(Y)  # list of unpickled traning data labels
        Xtr = np.concatenate(xs)  # merge unpickle data into a single whole data using concatenate
        Ytr = np.concatenate(ys)  # merge the labels as well
        del X, Y  # delete the unused variables to save space
        Xte, Yte = self.load_CIFAR_batch(os.path.join(self.data_path, 'test_batch'))  # test data from testing pickle batches
        return Xtr, Ytr, Xte, Yte


# generate h5 file
def generate_h5(Xtr, ytr, Xtst, ytst, filename):
    print("Generating H5 file...")
    hf = h5py.File(filename, 'w')
    hf.create_dataset('X_train', data=Xtr, compression="gzip")
    hf.create_dataset('y_train', data=ytr, compression="gzip")
    hf.create_dataset('X_test', data=Xtst, compression="gzip")
    hf.create_dataset('y_test', data=ytst, compression="gzip")
    hf.close()
    print("{} file generated".format(filename))


# function to split data containing images and labels into training and testing
def split_data(data, split):
    """
    this function takes in data and returns training and testing splits. since % of validation data varies based on
    applications we keep that parameter user configurable.

    :param data: list of lists; where each list is a [image, label] entity. eg: [image, 'dog'], [image, 'cat']
    :param train_split: percentage of training data to keep
    :param test_split: percentage of testing data to be kept
    :return:  return the splits
    """

    images = np.array([img[0] for img in data])  # unpacking images from the data list
    labels = np.array([img[1] for img in data])  # unpacking corresponding labels from the data list

    print("Number of images in the training data: {}".format(str(images.shape[0])))
    print("Labels: {}".format(str(labels.shape)))

    # multiply split percentage with total images length and floor the result. Also cast into int, for slicing array
    split_factor = int(np.floor(split * images.shape[0]))  # number of images to be kept in training data
    print("Using {} images for training and {} images for testing!".format(str(split_factor),
                                                                           str(images.shape[0] - split_factor)))
    x_train = images[:split_factor, :, :, :].astype("float")
    x_test = images[split_factor:, :, :, :].astype("float")
    y_train = labels[:split_factor]
    y_test = labels[split_factor:]

    print("Training data shape: {}".format(str(x_train.shape)))
    print("Testing data shape: {}".format(str(x_test.shape)))
    print("Training labels shape: {}".format(str(y_train.shape)))
    print("Testing labels shape: {}".format(str(y_test.shape)))

    return x_train, x_test, y_train, y_test


def data_from_dirs(data_dir, categories):
    """
    Citations: Python documentation was referred for understanding directory traversing using os
    :param filename: the current filename - in this case it point to this file -> signs1.py
    :return: current direcotory, train data and test data directory paths
    """
    data = list()  # a list of lists. Each tuple is a [image, label] format

    # loop over all the categories
    for index, category in enumerate(categories):
        path = os.path.join(data_dir, category)  # path to every alphabet
        # path name leading to the alphabet directory
        print("Opening directory {} from path: {}".format(str(category), str(path)))

        # loop over all the images in the directory
        for img in os.listdir(path):
            try:
                # reading the image in original format
                image = cv2.imread(os.path.join(path, img))
                data.append([image, index])  # append the [image, label] list in the data list
            except Exception as e:
                print(e)

    random.shuffle(data)  # randomly shuffling data

    return data


if __name__ == "__main__":
    # 26 alphabets, 3 special characters, total 29 unique characters, each having 3001 images, total = 3001 x 29 = 87029
    categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                  'space', 'nothing', 'del']
    asl_dir = "/Users/nikunjlad/data/asl/alphabets"  # path to the main alphabets directory
    data = data_from_dirs(asl_dir, categories)  # acquire all the data.
    x_train, y_train, x_test, y_test = split_data(data, 0.98)
    generate_h5(x_train, y_train, x_test, y_test, "alphabets.h5")
    sys.exit(1)
