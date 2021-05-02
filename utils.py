import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import metrics
from data_loader import CLASS_NAMES


def validate(nn, X, y, valid_batch_size=100):
    preds = []

    valid_batch_num = X.shape[0] // valid_batch_size
    if X.shape[0] % valid_batch_size != 0:
        valid_batch_num += 1

    for i in range(valid_batch_num):
        start_idx = i * valid_batch_size
        end_idx = (i + 1) * valid_batch_size
        scores = nn.predict(X[start_idx:end_idx])
        preds.append(np.argmax(scores, axis=1))

    preds = np.concatenate(preds, axis=None)
    correct_classified = np.count_nonzero(preds == y)
    acc = 100 * (correct_classified / len(y))  # calculate the accuracy
    print("%d/%d samples are correctly classified - Accuracy: %0.2f" % (correct_classified, len(y), acc))
    return acc


def plot_parameters(weight, size1, size2):
    plt.figure()
    plt.imshow(weight.reshape(size1, size2))
    plt.show()


def plot_loss(opt, loss):
    # plot_path = 'experiments/%dnn/%d/plots/' % (opt.hidden_layer_num, opt.hidden_unit_num)
    # plot_path = 'experiments/%dnn/vgg/plots/' % (opt.hidden_layer_num)
    # plot_path = 'experiments/slnn/vgg/plots/'
    # plt.title('Batch size=%d Learning rate=%0.3f' % (opt.batch_size, opt.learning_rate))
    plt.title('Objective=%s Activation=%s' % (opt.objective_func, opt.activation_func))
    plt.plot(np.arange(opt.epoch_num), loss, label='Train')
    plt.xticks(np.arange(1, opt.epoch_num, 2))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.savefig(plot_path + '%dnn-loss-obj=%s-act=%s.png' % (opt.hidden_layer_num, opt.objective_func, opt.activation_func))
    # plt.close()
    plt.show()


def plot_acc(opt, train, val):
    # plot_path = 'experiments/%dnn/%d/plots/' % (opt.hidden_layer_num, opt.hidden_unit_num)
    # plot_path = 'experiments/%dnn/vgg/plots/' % (opt.hidden_layer_num)
    # plot_path = 'experiments/slnn/vgg/plots/'
    # plt.title('Batch size=%d Learning rate=%0.3f' % (opt.batch_size, opt.learning_rate))
    plt.title('Objective=%s Activation=%s' % (opt.objective_func, opt.activation_func))
    x = np.arange(opt.epoch_num)
    plt.plot(x, train, label='Train')
    plt.plot(x, val, label='Validation')
    plt.xticks(np.arange(1, opt.epoch_num, 2))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.savefig(plot_path + '%dnn-acc-obj=%s-act=%s.png' % (opt.hidden_layer_num, opt.objective_func, opt.activation_func))
    # plt.close()
    plt.show()
    print('Best with Objective=%s Activation=%s: %0.2f' % (opt.objective_func, opt.activation_func, max(val)))


def write_file(opt, train_loss, train_acc, val_acc):
    # txt_path = 'experiments/%dnn/%d/' % (opt.hidden_layer_num, opt.hidden_unit_num)
    # txt_path = 'experiments/%dnn/vgg/' % (opt.hidden_layer_num)
    txt_path = 'experiments/slnn/vgg/'
    f = open(txt_path + '%dnn-acc-obj=%s-act=%s.txt' % (opt.hidden_layer_num, opt.objective_func, opt.activation_func),
             "a")
    f.write("Train Loss\n")
    f.write(str(train_loss) + "\n")
    f.write("Train Accuracy\n")
    f.write(str(train_acc) + "\n")
    f.write("Validation Accuracy\n")
    f.write(str(val_acc) + "\n")
    f.close()


def plot_conf_matrix(true, pred):
    conf_mat = metrics.confusion_matrix(true, pred)
    df_cm = pd.DataFrame(conf_mat, CLASS_NAMES, CLASS_NAMES)

    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='d', cmap=plt.get_cmap('jet'))  # font size
    plt.title("Confusion Matrix")
    plt.show()
