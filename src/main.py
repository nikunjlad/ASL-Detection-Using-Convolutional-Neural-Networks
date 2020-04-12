import torch, warnings, torchvision, os, h5py, time, yaml, datetime, logging
from utils.DataGen import DataGen
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
import json
from model import Net
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').disabled = True

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
        stats = dict()

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
        output_path = os.path.join("output/", self.config["DATALOADER"]["OUTPUT_PATH"])
        os.mkdir(output_path)

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

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(),
                              lr=self.config["HYPERPARAMETERS"]["LR"], momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_train_data_batches, eta_min=0)

        # training and validation loop
        epochs = self.config["HYPERPARAMETERS"]["EPOCHS"]
        history = list()
        train_start = time.time()
        best_val_loss = float('inf')

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

                scheduler.step()  # stepping through the learning rate for optimal convergence

                # if GPU mentioned.
                if self.train_on_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # Clean existing gradients
                optimizer.zero_grad()

                # Forward pass - compute outputs on input data using the model
                outputs = net(inputs)
                # self.logger.debug("Output of the model: {}".format(str(outputs)))
                # self.logger.debug("Output shape: {}".format(str(outputs.shape)))

                # Compute loss
                loss = criterion(outputs, labels)
                # self.logger.debug("Loss: {}".format(str(loss)))
                # self.logger.debug("Loss shape: {}".format(str(loss.shape)))

                # Backpropagate the gradients
                loss.backward()

                # Update the parameters
                optimizer.step()

                # Compute the total loss for the batch and add it to train_loss
                train_loss += loss.item() * inputs.size(0)
                # self.logger.debug(str(train_loss))

                # Compute the accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to train_acc
                train_acc += acc.item() * inputs.size(0)

                print("Batch: {:03d}/{:03d}, Training Loss: {:.4f}, "
                      "Training Acc: {:.4f}".format(i, num_train_data_batches, loss.item(), acc.item() * 100))

            # Validation - No gradient tracking needed
            with torch.no_grad():

                # Set to evaluation mode
                net.eval()

                # Validation loop
                for j, (inputs, labels) in enumerate(self.data["valid_dataloader"]):

                    if self.train_on_gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    # Forward pass - compute outputs on input data using the model
                    outputs = net(inputs)

                    # Compute loss
                    loss = criterion(outputs, labels)

                    # Compute the total loss for the batch and add it to valid_loss
                    valid_loss += loss.item() * inputs.size(0)

                    # Calculate validation accuracy
                    ret, predictions = torch.max(outputs.data, 1)
                    correct_counts = predictions.eq(labels.data.view_as(predictions))

                    # Convert correct_counts to float and then compute the mean
                    acc = torch.mean(correct_counts.type(torch.FloatTensor))

                    # Compute total accuracy in the whole batch and add to valid_acc
                    valid_acc += acc.item() * inputs.size(0)

                    print("Validation Batch number: {:03d}/{:03d}, Validation Loss: {:.4f}, "
                          "Validation Acc: {:.4f}".format(j, num_valid_data_batches, loss.item(), acc.item() * 100))

            # resetting scheduler
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_train_data_batches, eta_min=0)

            # Find average training loss and training accuracy
            avg_train_loss = train_loss / train_data_size
            avg_train_acc = train_acc / float(train_data_size)

            # Find average training loss and training accuracy
            avg_valid_loss = valid_loss / valid_data_size
            avg_valid_acc = valid_acc / float(valid_data_size)

            history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

            epoch_end = time.time()
            print("-" * 89)
            print("Epoch: {:03d}, Train Loss: {:.4f}, Train Acc: {:.4f}%, Valid Loss : {:.4f}, "
                  "Valid Acc: {:.4f}%, Time: {:.4f}s".format(epoch + 1, avg_train_loss, avg_train_acc * 100,
                                                             avg_valid_loss, avg_valid_acc * 100,
                                                             epoch_end - epoch_start))
            print("-" * 89)

            if avg_valid_loss < best_val_loss:
                print("\nPrevious Best loss: {:.4f} | New Best Loss: {:.4f} | "
                      "Saving Best model...\n".format(best_val_loss, avg_valid_loss))
                torch.save(net.state_dict(), output_path + "/" + self.config["DATALOADER"]["MODEL_PATH"])
                best_val_loss = avg_valid_loss  # new best loss is the recently found validation loss

        train_stop = time.time()
        self.logger.info("Time taken for training: {}".format(str(train_stop - train_start)))

        # saving model once training is done
        # torch.save(net.state_dict(), 'asl.pt')  # save the resnet model
        hist = np.array(history)  # convert history from list to numpy array

        # training and validation loss curves
        plt.figure(figsize=(7, 6))
        x = np.array([i for i in range(0, epochs)])
        plt.plot(x, hist[:, 0])
        plt.plot(x, hist[:, 1])
        plt.xlabel("Epochs")
        plt.ylabel("Cross-Entropy Loss")
        plt.title("CIFAR-10 Loss Curves")
        plt.legend(['train_loss', 'valid_loss'], loc='upper right')
        plt.savefig(output_path + "/train_valid_loss_3.png")

        # training and validation accuracy curves
        plt.figure(figsize=(7, 6))
        x = np.array([i for i in range(0, epochs)])
        plt.plot(x, hist[:, 2])
        plt.plot(x, hist[:, 3])
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("CIFAR-10 Accuracy Curves")
        plt.legend(['train_acc', 'valid_acc'], loc='upper right')
        plt.savefig(output_path + "/train_valid_accuracy_3.png")

        # load model after training for testing
        net.load_state_dict(torch.load(output_path + "/" + self.config["DATALOADER"]["MODEL_PATH"]))

        test_loss = 0
        test_acc = 0
        test_hist = list()

        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            net.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(self.data["test_dataloader"]):
                inputs = inputs.cuda()
                labels = labels.cuda()

                # Forward pass - compute outputs on input data using the model
                outputs = net(inputs)

                # Compute loss
                loss = criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                test_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                print(predictions.cpu().numpy()[0])
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                test_acc += acc.item() * inputs.size(0)

                print("Test Batch number: {:03d}/{:03d}, Test Loss: {:.4f}, "
                      "Test Accuracy: {:.4f}".format(j, num_test_data_batches, loss.item(), acc.item() * 100))

            avg_test_loss = test_loss / test_data_size
            avg_test_acc = test_acc / float(test_data_size)

            test_hist.append([avg_test_loss, avg_test_acc])

            print("Test: Loss : {:.4f}, Accuracy: {:.4f}%".format(avg_test_loss, avg_test_acc * 100))

        # writing stats

        stats["hyperparameters"] = dict()
        stats["device"] = dict()
        stats["device"]["type"] = ["gpu" if self.config["HYPERPARAMETERS"]["GPU"] else "cpu"][0]
        stats["device"]["parallel"] = self.config["HYPERPARAMETERS"]["PARALLEL"]
        stats["device"]["devices"] = list()
        if stats["device"]["type"] == "gpu":
            if isinstance(self.config["HYPERPARAMETERS"]["DEVICES"], list):
                for dev in self.config["HYPERPARAMETERS"]["DEVICES"]:
                    info = dict()
                    info["id"] = dev
                    info["device_name"] = torch.cuda.get_device_properties(dev).name
                    info["total_memory (MB)"] = torch.cuda.get_device_properties(dev).total_memory * (2 ** -20)
                    stats["device"]["devices"].append(info)
            else:
                info = dict()
                info["id"] = self.config["HYPERPARAMETERS"]["DEVICES"][0]
                info["device_name"] = torch.cuda.get_device_properties(info["id"]).name
                info["total_memory (MB)"] = torch.cuda.get_device_properties(info["id"]).total_memory * (2 ** -20)
                stats["device"]["devices"].append(info)
        stats["hyperparameters"]["epochs"] = self.config["HYPERPARAMETERS"]["EPOCHS"]
        stats["hyperparameters"]["learning_rate"] = self.config["HYPERPARAMETERS"]["LR"]
        stats["hyperparameters"]["batch_size"] = self.config["HYPERPARAMETERS"]["BATCH_SIZE"]
        stats["hyperparameters"]["optimizer"] = self.config["HYPERPARAMETERS"]["OPTIMIZER"]
        stats["hyperparameters"]["activation"] = self.config["HYPERPARAMETERS"]["ACTIVATION"]
        if stats["device"]["parallel"]:
            stats["hyperparameters"]["dropout"] = net.module.dropout.p
        else:
            stats["hyperparameters"]["dropout"] = net.dropout.p
        stats["metrics"] = dict()
        stats["metrics"]["training_loss"] = history[-1][0]
        stats["metrics"]["training_accuracy"] = history[-1][2]
        stats["metrics"]["validation_loss"] = history[-1][1]
        stats["metrics"]["validation_accuracy"] = history[-1][3]
        stats["metrics"]["test_loss"] = test_hist[-1][0]
        stats["metrics"]["test_accuracy"] = test_hist[-1][1]
        stats["metrics"]["runtime (secs)"] = train_stop - train_start
        stats["training_history"] = history
        stats["test_history"] = test_hist

        with open(output_path + "/" + self.config["DATALOADER"]["STATS_PATH"], 'w') as outfile:
            json.dump(stats, outfile, indent=2)

        print(stats)


if __name__ == '__main__':
    m = Main()
    m.main()
