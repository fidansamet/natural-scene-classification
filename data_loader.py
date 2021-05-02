import os
import random
import torch
import csv
import re
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import vgg19

TRAIN_PATH = '/seg_train/seg_train/'
VALID_PATH = '/seg_dev/seg_dev/'
TEST_PATH = '/seg_test/'
VGG19_PATH = 'vgg19_features/'
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
RANDOM_SEED = 42
NEW_IMG_SIZE = (30, 30)


class DataLoader:
    def __init__(self, opt):
        self.opt = opt

    def process_img(self, img, data):
        if self.opt.vgg19:
            return self.extract_vgg_features(img, data)
        else:
            resized_img = img.resize(NEW_IMG_SIZE)  # resize image to 30x30
            gray_img = ImageOps.grayscale(resized_img)  # convert image to grayscale
            flatten = np.asarray(gray_img).flatten()  # flatten the matrix
            return flatten / 255.
            # return (flatten - min(flatten)) / (max(flatten) - min(flatten))

    def load_labeled_data(self, path, data):
        X, y = [], []
        for i in range(len(CLASS_NAMES)):
            cur_path = self.opt.data_path + path + CLASS_NAMES[i]
            for img_name in sorted(os.listdir(cur_path)):
                img = Image.open(cur_path + '/' + img_name)
                X.append(self.process_img(img, data))
                y.append(i)
        # shuffle data
        zipped = list(zip(X, y))
        random.Random(RANDOM_SEED).shuffle(zipped)
        X, y = zip(*zipped)
        return np.asarray(X), np.asarray(y)

    def load_train_and_valid(self):
        print('Train and validation data load started.')

        if self.opt.vgg19:
            self.get_vgg_features('train')
            self.get_vgg_features('valid')

        self.X_train, self.y_train = self.load_labeled_data(TRAIN_PATH, 'train')
        self.X_valid, self.y_valid = self.load_labeled_data(VALID_PATH, 'valid')

        if self.opt.vgg19:
            self.save_vgg_features('train')
            self.save_vgg_features('valid')

        print('Train and validation data load ended.')

    def load_test(self):
        print('Test data load started.')

        if self.opt.vgg19:
            self.get_vgg_features('test')
        self.X_test, self.y_test = [], []

        cur_path = self.opt.data_path + TEST_PATH  # TODO
        img_paths = os.listdir(cur_path)
        img_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
        csv_reader = csv.reader(open(self.opt.test_label_path), delimiter=';')
        next(csv_reader)  # discard header

        for img_name, label_row in zip(img_paths, csv_reader):
            img = Image.open(cur_path + '/' + img_name)
            self.X_test.append(self.process_img(img, 'test'))
            self.y_test.append(int(label_row[1]))
        self.X_test = np.asarray(self.X_test)
        self.y_test = np.asarray(self.y_test)

        if self.opt.vgg19:
            self.save_vgg_features('test')

        print('Test data load ended.')

    # VGG19
    def get_vgg_features(self, data):
        if os.path.exists(VGG19_PATH + data + '.txt'):
            if data == 'train':
                self.vgg19_train_read = True
                self.vgg19_train_iter = iter(np.loadtxt(VGG19_PATH + data + '.txt', dtype=np.float32))
            elif data == 'valid':
                self.vgg19_valid_read = True
                self.vgg19_valid_iter = iter(np.loadtxt(VGG19_PATH + data + '.txt', dtype=np.float32))
            else:
                self.vgg19_test_read = True
                self.vgg19_test_iter = iter(np.loadtxt(VGG19_PATH + data + '.txt', dtype=np.float32))
        else:
            open(VGG19_PATH + data + '.txt', 'w')
            self.vgg19 = VGG19()
            if data == 'train':
                self.vgg19_train_read = False
                self.vgg19_train_features = []
            elif data == 'valid':
                self.vgg19_valid_read = False
                self.vgg19_valid_features = []
            else:
                self.vgg19_test_read = False
                self.vgg19_test_features = []

    def extract_vgg_features(self, img, data):
        if data == 'train' and self.vgg19_train_read:
            vgg19_extracted = next(self.vgg19_train_iter)
        elif data == 'valid' and self.vgg19_valid_read:
            vgg19_extracted = next(self.vgg19_valid_iter)
        elif data == 'test' and self.vgg19_test_read:
            vgg19_extracted = next(self.vgg19_test_iter)
        else:
            pil_image = img.convert("RGB")
            vgg19_extracted = self.vgg19.forward(pil_image)
            vgg19_extracted = vgg19_extracted.numpy()[0]

            if data == 'train':
                self.vgg19_train_features.append(vgg19_extracted)
            elif data == 'valid':
                self.vgg19_valid_features.append(vgg19_extracted)
            else:
                self.vgg19_test_features.append(vgg19_extracted)
        return vgg19_extracted

    def save_vgg_features(self, data):
        if data == 'train' and not self.vgg19_train_read:
            np.savetxt(VGG19_PATH + 'train.txt', self.vgg19_train_features, fmt='%f')
        elif data == 'valid' and not self.vgg19_valid_read:
            np.savetxt(VGG19_PATH + 'valid.txt', self.vgg19_valid_features, fmt='%f')
        elif data == 'test' and not self.vgg19_test_read:
            np.savetxt(VGG19_PATH + 'test.txt', self.vgg19_test_features, fmt='%f')


class VGG19:
    def __init__(self):
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = vgg19(pretrained=True)
        # change mode to evaluation
        self.model.eval()
        # remove last fully connected layer
        self.model.classifier = torch.nn.Sequential(*list(self.model.classifier.children())[:-1])

    def forward(self, img):
        # preprocess image for network
        input_tensor = self.preprocess(img)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            features = self.model(input_batch)
        return features
