import torch, warnings, torchvision, os, h5py, time, yaml, datetime, logging
from utils.DataGen import DataGen
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image
import matplotlib
import torch.backends.cudnn as cudnn
from tqdm import tqdm
matplotlib.use("TkAgg")
from model import Net
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

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
            os.mkdir("logs")  # make log directory if does not exist
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
        logger = logging.getLogger()  # name the logger as asl
        logger.setLevel(logging.DEBUG)
        f_hand = logging.FileHandler(name)  # file where the custom logs needs to be handled
        f_hand.setLevel(logging.DEBUG)  # level to set for logging the errors
        f_format = logging.Formatter('%(asctime)s : %(process)d : %(levelname)s : %(message)s',
                                     datefmt='%d-%b-%y %H:%M:%S')  # format in which the logs needs to be written
        f_hand.setFormatter(f_format)  # setting the format of the logs
        logger.addHandler(f_hand)  # setting the logging handler with the above formatter specification

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(f_format)
        logger.addHandler(ch)

        return logger

    # checking if cuda is available
    def configure_cuda(self, device_id):
        self.train_on_gpu = torch.cuda.is_available()
        if not self.train_on_gpu:
            self.logger.info('CUDA is not available. Training on CPU ...')
        else:
            torch.cuda.set_device(device_id)
            self.logger.info(
                'CUDA is available! Training on Tesla T4 Device {}'.format(str(torch.cuda.current_device())))

    def main(self):
        """
        This is the wrapper function which calls and generates data from other classes and helper functions
        :return: main program execution
        """

        # configure GPU if available
        if self.config["HYPERPARAMETERS"]["GPU"]:
            if self.config["HYPERPARAMETERS"]["DEVICES"] is not None:
                self.configure_cuda(self.config["HYPERPARAMETERS"]["DEVICES"][0])

        # configure data path
        if os.getenv("HOME") != self.config["DATALOADER"]["DATA_DIR"]:
            self.config["DATALOADER"]["DATA_DIR"] = os.getenv("HOME")

        classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                   'space', 'nothing', 'del']

        # loading data
        data_path = os.path.join(self.config["DATALOADER"]["DATA_DIR"], "data/asl/alphabets.h5")

        self.load_data_from_h5(data_path)
        self.split_data()
        self.configure_dataloaders()

        # get training, validation and testing dataset sizes and number of batches in each
        train_data_size = len(self.data["train_dataset"])
        valid_data_size = len(self.data["valid_dataset"])
        test_data_size = len(self.data["test_dataset"])
        num_train_data_batches = len(self.data["train_dataloader"])
        num_valid_data_batches = len(self.data["valid_dataloader"])
        num_test_data_batches = len(self.data["test_dataloader"])

        # display batch information
        self.logger.info("Number of training samples: {}".format(str(train_data_size)))
        self.logger.info("{} batches each having 64 samples".format(str(num_train_data_batches)))
        self.logger.info("Number of validation samples: {}".format(str(valid_data_size)))
        self.logger.info("{} batches each having 64 samples".format(str(num_valid_data_batches)))
        self.logger.info("Number of testing samples: {}".format(str(test_data_size)))
        self.logger.info("{} batches each having 64 samples".format(str(num_test_data_batches)))

        # export a subset of images
        batch = next(iter(self.data["test_dataloader"]))
        images, labels = batch

        if self.config["HYPERPARAMETERS"]["PLOT_IMG"]:
            grid = torchvision.utils.make_grid(images[:64], nrow=8)
            self.logger.debug(type(grid))
            plt.figure(figsize=(10, 10))
            np.transpose(grid, (1, 2, 0))
            save_image(grid, 'grid.png')
            for data, target in self.data["test_dataloader"]:
                self.logger.debug("Batch image tensor dimensions: {}".format(str(data.shape)))
                self.logger.debug("Batch label tensor dimensions: {}".format(str(target.shape)))
                break

        net = Net()
        if self.train_on_gpu:
            net = net.cuda()
        self.logger.debug(str(net))

        if self.config["HYPERPARAMETERS"]["PARALLEL"]:
            net = torch.nn.DataParallel(net, device_ids=self.config["HYPERPARAMETERS"]["DEVICES"])
            cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(),
                              lr=self.config["HYPERPARAMETERS"]["LR"], momentum=0.9, weight_decay=5e-4)

        # training and validation loop
        epochs = self.config["HYPERPARAMETERS"]["EPOCHS"]
        history = list()
        train_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()  # start time for the epoch
            print("Epoch: {}/{}".format(epoch + 1, epochs))

            # Set to training mode
            net.train()

            # Loss and Accuracy within the epoch
            train_loss = 0.0
            train_acc = 0.0

            valid_loss = 0.0
            valid_acc = 0.0

            for i, (inputs, labels) in enumerate(self.data["train_dataloader"]):

                # if GPU mentioned.
                if self.train_on_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # Clean existing gradients
                optimizer.zero_grad()

                # Forward pass - compute outputs on input data using the model
                outputs = net(inputs)
                self.logger.debug(str(outputs))

                # Compute loss
                loss = criterion(outputs, labels)
                self.logger.debug(str(loss))

                # Backpropagate the gradients
                loss.backward()

                # Update the parameters
                optimizer.step()

                # Compute the total loss for the batch and add it to train_loss
                train_loss += loss.item() * inputs.size(0)
                self.logger.debug(str(train_loss))

                # Compute the accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to train_acc
                train_acc += acc.item() * inputs.size(0)

                print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(),
                                                                                              acc.item() * 100))

                break
            break


if __name__ == '__main__':
    m = Main()
    m.main()
