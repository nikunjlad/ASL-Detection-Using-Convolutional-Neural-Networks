from sklearn.metrics import log_loss, roc_auc_score, roc_curve, precision_recall_curve, \
    confusion_matrix, f1_score
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_confusion_matrix


def mean_squared_error(history, plot):
    mse = history.history['mean_squared_error']

    if plot:
        plt.plot(mse)
        plt.title('Mean Squared Error')
        plt.show()
    return mse


def mean_absolute_error(history, plot):
    mae = history.history['mean_absolute_error']

    if plot:
        plt.plot(mae)
        plt.title('Mean Absolute Error')
        plt.show()

    return mae


def mean_absolute_percentage_error(history, plot):
    mape = history.history['mean_absolute_percentage_error']

    if plot:
        plt.plot(mape)
        plt.title('Mean Absolute Percentage Error')
        plt.show()
    return mape


def l_loss(actual, predicted, plot):
    loss = log_loss(actual, predicted)

    if plot:
        plt.plot(predicted, loss)
        plt.title("Log Loss / Cross Entropy")
        plt.show()
    return loss


def train_validation_accuracy(history, plot):
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']

    if plot:
        plt.plot(train_acc)
        plt.plot(val_acc)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    return train_acc, val_acc


def train_validation_loss(history, plot):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    if plot:
        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    return train_loss, val_loss


def confusion(actual, predicted, plot):
    result = confusion_matrix(actual, predicted)

    if plot:
        _, __ = plot_confusion_matrix(conf_mat=result, show_absolute=True, show_normed=True, colorbar=True)
        plt.show()


def f_measure(actual, predicted):
    f1 = f1_score(actual, predicted, average="weighted")

    return f1


def precision_recall(actual, predicted, plot):
    precision, recall, _ = precision_recall_curve(actual, predicted)

    if plot:
        plt.plot([0, 1], [0.1, 0.1], linestyle='--')
        # plot the roc curve for the model
        plt.plot(recall, precision, marker='.')
        # show the plot
        plt.show()

    return precision, recall


def auc_roc(actual, predicted, plot):
    auc = roc_auc_score(actual, predicted)

    fpr, tpr, thresholds = roc_curve(actual, predicted)

    if plot:
        plt.plot([0, 1], [0, 1], linestyle='--')
        # plot the roc curve for the model
        plt.plot(fpr, tpr, marker='.')
        # show the plot
        plt.show()

    return auc, fpr, tpr
