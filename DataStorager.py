import torch
import cv2
import os
import matplotlib.pyplot as plt


class DataStorage:
    def __init__(self):
        self.data_train = torch.Tensor()
        self.data_test = torch.Tensor()
        self.labels_train = torch.Tensor()
        self.labels_test = torch.Tensor()

    def load_data_train(self):
        files_diana = list(map(lambda x: "diana/" + x, os.listdir(path='data/train/diana')))
        files_not_diana = list(map(lambda x: "not_diana/" + x, os.listdir(path='data/train/not_diana')))
        files = files_diana + files_not_diana
        self.data_train = [cv2.imread('data/train/' + path) for path in files]

        diana_size = len(files_diana)
        not_diana_size = len(files_not_diana)
        self.labels_train = torch.cat(
            (torch.ones(diana_size, dtype=torch.long), torch.zeros(not_diana_size, dtype=torch.long)))

    def load_data_test(self):
        files_diana_test = list(map(lambda x: "diana/" + x, os.listdir(path='data/test/diana')))
        files_not_diana_test = list(map(lambda x: "not_diana/" + x, os.listdir(path='data/test/not_diana')))
        files_test = files_diana_test + files_not_diana_test
        self.data_test = [cv2.imread('data/test/' + path) for path in files_test]

        diana_size_test = len(files_diana_test)
        not_diana_size_test = len(files_not_diana_test)
        self.labels_test = torch.cat(
            (torch.ones(diana_size_test, dtype=torch.long), torch.zeros(not_diana_size_test, dtype=torch.long)))

    def prepare_dataset(self):
        self.data_train = [self.process_image(image) for image in self.data_train]
        self.data_test = [self.process_image(image) for image in self.data_test]
        self.data_train = torch.tensor(self.data_train, dtype=torch.float)
        self.data_test = torch.tensor(self.data_test, dtype=torch.float)
        self.data_train = self.data_train.permute(0, 3, 1, 2)
        self.data_test = self.data_test.permute(0, 3, 1, 2)
        perm = torch.randperm(self.data_train.shape[0])
        self.data_train = self.data_train[perm]
        self.labels_train = self.labels_train[perm]
        print(self.data_train.shape)
        print(self.data_test.shape)

    @staticmethod
    def process_image(image):
        image = cv2.resize(image, (256, 256))
        image = image / 255
        return image




