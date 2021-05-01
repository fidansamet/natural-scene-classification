import numpy as np
import random
from data_loader import DataLoader
from options import Options
from neural_network import NeuralNetwork

np.random.seed(12345)
loss_cache, train_acc_cache, valid_acc_cache = [], [], []
lr_decay = 0.95


def validate(X, y, valid_batch_size=100):
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


def mini_batch_gd(start_idx, end_idx):
    # prepare mini batches of data
    # random_batch = np.random.choice(X_train.shape[0], batch_size)
    X_batch = X_train[start_idx:end_idx]
    y_batch = y_train[start_idx:end_idx]

    loss, gradients = nn.train(X_batch, y_batch)        # train network with batches
    loss_cache.append(loss)
    nn.update_weights(gradients)    # update parameters
    return loss


if __name__ == '__main__':
    opt = Options().parse()
    data_loader = DataLoader(opt)
    data_loader.load_train_and_valid()
    hidden_sizes = [opt.hidden_unit_num for i in range(opt.hidden_layer_num)]
    input_size = 4098 if opt.vgg19 else 900
    nn = NeuralNetwork(hidden_sizes=hidden_sizes, activation_func=opt.activation_func, error_func=opt.objective_func,
                       lr=opt.learning_rate, input_size=input_size)

    X_train, y_train = data_loader.X_train, data_loader.y_train
    X_valid, y_valid = data_loader.X_valid, data_loader.y_valid
    batch_size = opt.batch_size
    epoch_num = opt.epoch_num
    batch_num = X_train.shape[0] // batch_size

    for epoch in range(epoch_num):
        zipped = list(zip(X_train, y_train))
        random.Random().shuffle(zipped)
        X_train, y_train = zip(*zipped)
        X_train, y_train = np.asarray(X_train), np.asarray(y_train)

        for i in range(batch_num):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            loss = mini_batch_gd(start_idx, end_idx)
            # print('Iteration %d in Epoch %d - Loss: %f' % (i+1, epoch+1, loss))

        if opt.reduce_lr:
            nn.lr *= lr_decay

        train_acc = validate(X_train, y_train)
        train_acc_cache.append(train_acc)
        print('Epoch %d/%d - Train acc: %f' % (epoch+1, epoch_num, train_acc))

        valid_acc = validate(X_valid, y_valid)
        valid_acc_cache.append(valid_acc)
        print('Epoch %d/%d - Validation acc: %f' % (epoch+1, epoch_num, valid_acc))

        print("-------------------")
        # nn.save_model()
