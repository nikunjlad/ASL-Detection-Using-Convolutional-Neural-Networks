import torch, warnings, torchvision, os, h5py, time, yaml, datetime, logging
from metrics import *
from utils.DataGen import DataGen

# only A and B
categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              'space', 'nothing', 'del']


class Main(DataGen):

    def __init__(self):
        # loading the YAML configuration file
        with open("config.yaml", 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d  %H.%M")  # get current datetime
        if not os.path.exists("logs"):
            os.mkdir("logs")   # make log directory if does not exist
        os.chdir("logs")  # change to logs directory
        # getting the custom logger
        self.logger_name = "asl_" + self.current_time + "_.log"
        self.logger = self.get_loggers(self.logger_name)
        self.logger.info("ASL Detection!")
        self.logger.info("Current time: " + str(self.current_time))
        self.train_on_gpu = False
        DataGen.__init__(self, self.config, self.logger)
        os.chdir("..")  # change directory to base path

    @staticmethod
    def get_loggers(name):
        logger = logging.getLogger("asl")  # name the logger as asl
        logger.setLevel(logging.DEBUG)
        f_hand = logging.FileHandler(name)     # file where the custom logs needs to be handled
        f_hand.setLevel(logging.INFO)        # level to set for logging the errors
        f_format = logging.Formatter('%(asctime)s : %(process)d : %(levelname)s : %(message)s',
                                     datefmt='%d-%b-%y %H:%M:%S')  # format in which the logs needs to be written
        f_hand.setFormatter(f_format)  # setting the format of the logs
        logger.addHandler(f_hand)  # setting the logging handler with the above formatter specification

        return logger

    # checking if cuda is available
    def configure_cuda(self, device_id):
        self.train_on_gpu = torch.cuda.is_available()
        if not self.train_on_gpu:
            print('CUDA is not available. Training on CPU ...')
        else:
            torch.cuda.set_device(device_id)
            print('CUDA is available! Training on Tesla T4 Device {}'.format(str(torch.cuda.current_device())))

    # def data_dirs(filename):
    #     """
    #     Citations: Python documentation was referred for understanding directory traversing using os
    #     :param filename: the current filename - in this case it point to this file -> signs1.py
    #     :return: current direcotory, train data and test data directory paths
    #     """
    #     current_dir = os.path.dirname(os.path.abspath(filename))
    #     os.chdir('../data')
    #     data_dir = os.getcwd()
    #     train_data_dir = data_dir + '/asl_alphabet_train/'
    #     test_data_dir = data_dir + '/asl_alphabet_test/'
    #
    #     return current_dir, train_data_dir, test_data_dir

    def main(self):
        """
        This is the wrapper function which calls and generates data from other classes and helper functions
        :return: does specifically return anything.
        """

        # configure GPU if available
        if self.config["HYPERPARAMETERS"]["GPU"]:
            if self.config["HYPERPARAMETERS"]["DEVICES"] is not None:
                self.configure_cuda(self.config["HYPERPARAMETERS"]["DEVICES"][0])

        # configure data path
        if os.getenv("HOME") != self.config["DATALOADER"]["DATA_DIR"]:
            self.config["DATALOADER"]["DATA_DIR"] = os.getenv("HOME")

        # loading data
        data_path = os.path.join(self.config["DATALOADER"]["DATA_DIR"], "data/asl/alphabets.h5")
        self.load_data(data_path)
        self.split_data()
        self.configure_dataloaders()

        classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                      'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                      'space', 'nothing', 'del']

        # get training, validation and testing dataset sizes and number of batches in each
        train_data_size = len(self.data["train_dataset"])
        valid_data_size = len(self.data["valid_dataset"])
        test_data_size = len(self.data["test_dataset"])
        num_train_data_batches = len(self.data["train_dataloader"])
        num_valid_data_batches = len(self.data["valid_dataloader"])
        num_test_data_batches = len(self.data["test_dataloader"])

        # display batch information
        self.logger.log("Number of training samples: {}".format(str(train_data_size)))
        self.logger.log("{} batches each having 64 samples".format(str(num_train_data_batches)))
        self.logger.log("Number of validation samples: {}".format(str(valid_data_size)))
        self.logger.log("{} batches each having 64 samples".format(str(num_valid_data_batches)))
        self.logger.log("Number of testing samples: {}".format(str(test_data_size)))
        self.logger.log("{} batches each having 64 samples".format(str(num_test_data_batches)))

        # # get current and data directories (train and test)
        # current_dir, train_data_dir, test_data_dir = data_dirs(__file__)
        #
        # # create an object of CNN and use it to create training and testing data along with their corresponding labels
        # cnn = CNN(categories, train_data_dir, test_data_dir)
        # train_matrix, train_labels = cnn.create_training_data()
        # test_matrix, test_labels = cnn.create_testing_data()
        #
        # # resize images with the earlier defined size and pass in the matrices to be resized
        # train_matrix = cnn.resize_images(Img_Size[0], Img_Size[1], train_matrix)
        # test_matrix = cnn.resize_images(Img_Size[0], Img_Size[1], test_matrix)
        #
        # # convert data to numpy array
        # train_data = cnn.create_numpy_data(train_matrix)
        # test_data = cnn.create_numpy_data(test_matrix)
        #
        # # reshaping the images in order to make it ready for convolution
        # input_shape, train_data, __, _ = cnn.data_reshape(train_data, train_labels, 1)
        # ___, test_data, classes, nClasses = cnn.data_reshape(test_data, test_labels, 1)
        #
        # # convert into float32 type
        # train_data = cnn.change_datatype(train_data, 'float32')
        # test_data = cnn.change_datatype(test_data, 'float32')
        #
        # # scaling data
        # train_data = cnn.scale_images(train_data, 255)
        # test_data = cnn.scale_images(test_data, 255)
        #
        # # one hot encoding labels
        # train_labels = cnn.do_one_hot_encoding(train_labels)
        # test_labels = cnn.do_one_hot_encoding(test_labels)
        #
        # # create CNN model. we have 32 filters in each layer, with activation function as sigmoid, along with filter size as
        # model = cnn.create_model(input_shape, activation="sigmoid", no_of_filters=32, filter_size=3, pool_size=2,
        #                          dense_layers=2)
        #
        # # compile model with setting these hyper parameters
        # model = cnn.compile_model(model=model, loss="binary_crossentropy", optimizer="adam",
        #                           metrics=['accuracy', 'mse', 'mae', 'cosine', 'mape'])
        #
        # # fit model and generate history
        # history = cnn.fit_model(model=model, train_data=train_data, train_labels=train_labels, batch_size=30, epochs=2,
        #                         verbose=1, splits=0.3)
        # print("History", history)
        #
        # # find out model scores
        # scores = cnn.evaluate_model(model=model, test_data=test_data, test_labels=test_labels)
        # print("Scores", scores)
        #
        # # find testing accuracy and predicted values
        # predicted_values = cnn.predict_using_model(model=model, test_data=test_data, test_labels=test_labels)
        # print(predicted_values)
        #
        # ## Estimate the model performance using the following metrics
        #
        # mse = mean_squared_error(history, True)
        # print(mse)
        #
        # mae = mean_absolute_error(history, True)
        # print(mae)
        #
        # mape = mean_absolute_percentage_error(history, True)
        # print(mape)
        #
        # train_acc, val_acc = train_validation_accuracy(history, True)
        #
        # train_loss, val_loss = train_validation_loss(history, True)
        # print(train_acc, val_acc, train_loss, val_loss)
        #
        # confusion(test_labels.argmax(axis=1), predicted_values.round().argmax(axis=1), True)
        #
        # # lloss = l_loss(test_labels, predicted_values, True)
        # # print(lloss)
        # # print("Test",test_labels)
        # # print("Predicted",predicted_values)
        # # f1 = f1_score(test_labels, predicted_values)
        # # print(f1)
        #
        # # r = auc_roc(test_labels, predicted_values, True)


if __name__ == '__main__':
    m = Main()
    m.main()
