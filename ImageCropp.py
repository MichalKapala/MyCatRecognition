import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

class ImageStorager:
    def __init__(self):
        self.picture = None
        self.width = 0
        self.height = 0
        self.depth = 0
        self.path = ""

    def load_image(self, path):
        self.path = path
        self.picture = cv2.imread(path)
        self.picture = cv2.resize(self.picture, (800, 800))
        self.width, self.height, self.depth = self.picture.shape

    def get_blob_image(self):
        blob_image = cv2.dnn.blobFromImage(self.picture, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        return blob_image


class Cropp:
    def __init__(self):
        self.model = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolo.cfg")
        self.layers_names = self.model.getLayerNames()
        self.outputlayers = []
        self.classes = []
        self.image = None
        self.picture = None

        self.set_output_layers()
        self.set_classes()

    def set_image(self, image):
        self.image = image
        self.picture = image.picture

    def set_output_layers(self):
        self.outputlayers = [self.layers_names[i[0] - 1]
                             for i in self.model.getUnconnectedOutLayers()]

    def set_classes(self):
        with open("yolo/coco.names", "r") as file:
            self.classes = [line.strip() for line in file.readlines()]

    def detect(self, blob_image):
        self.model.setInput(blob_image)
        outputs = self.model.forward(self.outputlayers)
        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and self.classes[class_id] == 'cat':
                    center_x = int(detection[0] * self.image.width)
                    center_y = int(detection[1] * self.image.height)
                    w = int(detection[2] * self.image.width)
                    h = int(detection[3] * self.image.height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    cropped_image = self.cropp(x, y, x+w, y+h)
                    self.save_cropped(cropped_image)

    def cropp(self, x_start, y_start, x_end, y_end):
        x_start = self.correct_x(x_start)
        x_end = self.correct_x(x_end)
        y_start = self.correct_y(y_start)
        y_end = self.correct_y(y_end)
        print(x_start,x_end,y_start, y_end)
        picture = self.picture[y_start:y_end, x_start:x_end]
        print(picture.shape)

        return picture

    def save_cropped(self, cropped_image):
        cv2.imwrite(self.image.path, cropped_image)

    def correct_x(self, coord):
        if coord < 0:
            coord = 0
        elif coord > self.image.width:
            coord = self.image.width
        return coord

    def correct_y(self, coord):
        if coord < 0:
            coord = 0
        elif coord > self.image.height:
            coord = self.image.height
        return coord

def main():
    files_diana = list(map(lambda x: "not_diana/" + x, os.listdir(path='data/test/not_diana')))
    for file in files_diana:
        model = Cropp()
        file = 'data/test/' + file
        image_file = ImageStorager()
        image_file.load_image(file)
        model.set_image(image_file)
        blob_img = image_file.get_blob_image()
        model.detect(blob_img)


if __name__ == '__main__':
    main()
