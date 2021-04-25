import os
import random
import numpy as np
from PIL import Image, ImageOps

TRAIN_PATH = '/seg_train/seg_train/'
VALID_PATH = '/seg_dev/seg_dev/'
TEST_PATH = '/seg_test/'
SUBSET_DIR_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
RANDOM_SEED = 42
NEW_IMG_SIZE = (30, 30)


class DataLoader:
    def __init__(self, opt):
        self.opt = opt

    def process_img(self, img):
        resized_img = img.resize(NEW_IMG_SIZE)  # resize image to 30x30
        gray_img = ImageOps.grayscale(resized_img)  # convert image to grayscale
        return np.asarray(gray_img).flatten()  # flatten the matrix

    def load_labeled_data(self, path):
        X, y = [], []
        for i in range(len(SUBSET_DIR_NAMES)):
            cur_path = self.opt.data_path + path + SUBSET_DIR_NAMES[i]
            for img_name in sorted(os.listdir(cur_path)):
                img = Image.open(cur_path + '/' + img_name)
                X.append(self.process_img(img))
                y.append(i)
        # shuffle data
        zipped = list(zip(X, y))
        random.Random(RANDOM_SEED).shuffle(zipped)
        X, y = zip(*zipped)
        return X, y

    def load_train_and_valid(self):
        print('Train and validation data load started.')
        self.X_train, self.y_train = self.load_labeled_data(TRAIN_PATH)
        self.X_valid, self.y_valid = self.load_labeled_data(VALID_PATH)
        print('Train and validation data load ended.')

    def load_test(self):
        print('Test data load started.')
        self.X_test = []
        cur_path = self.opt.data_path + TEST_PATH
        for img_name in sorted(os.listdir(cur_path)):
            img = Image.open(cur_path + '/' + img_name)
            self.X_test.append(self.process_img(img))
        print('Test data load ended.')
