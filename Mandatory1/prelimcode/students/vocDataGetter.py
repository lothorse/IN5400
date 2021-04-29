from vocparseclslabels import PascalVOC
import pandas as pd
import os
import numpy as np
import random

np.random.seed(420)
random.seed(420)

class DataGetter(PascalVOC):

    def trainingSettWithLabels(self, shuffle=True):
        filename = os.path.join(self.root_dir, "ImageSets", "Main", "train.txt")

        df = pd.read_csv(
            filename,
            header=None,
            names=['filename'])

        imgList = list(df['filename'].values)

        if shuffle:
            random.shuffle(imgList)

        categories = self.list_image_sets()

        oneHotCategories = np.zeros((len(imgList), len(categories)))
        homeless_images = []
        for i in range(len(categories)):
            cat = categories[i]
            imgs = self.imgs_from_category_as_list(cat, "train")
            for img in imgs:
                oneHotCategories[imgList.index(img)][i] = 1

        return imgList, oneHotCategories

    def valSettWithLabels(self, shuffle=False):
        filename = os.path.join(self.root_dir, "ImageSets", "Main", "val.txt")

        df = pd.read_csv(
            filename,
            header=None,
            names=['filename'])

        imgList = list(df['filename'].values)

        if shuffle:
            random.shuffle(imgList)

        categories = self.list_image_sets()

        oneHotCategories = np.zeros((len(imgList), len(categories)))
        homeless_images = []
        for i in range(len(categories)):
            cat = categories[i]
            imgs = self.imgs_from_category_as_list(cat, "val")
            for img in imgs:
                oneHotCategories[imgList.index(img)][i] = 1

        return imgList, oneHotCategories

    def testSettWithLabels(self, shuffle=False):
        filename = os.path.join(self.root_dir, "ImageSets", "Main", "test.txt")

        df = pd.read_csv(
            filename,
            header=None,
            names=['filename'])

        imgList = list(df['filename'].values)

        if shuffle:
            random.shuffle(imgList)

        categories = self.list_image_sets()

        oneHotCategories = np.zeros((len(imgList), len(categories)))
        homeless_images = []
        for i in range(len(categories)):
            cat = categories[i]
            imgs = self.imgs_from_category_as_list(cat, "test")
            for img in imgs:
                oneHotCategories[imgList.index(img)][i] = 1

        return imgList, oneHotCategories

    def all_img_names(self):
        filename = os.path.join(self.root_dir, "ImageSets", "Main", "trainval.txt")

        df = pd.read_csv(
            filename,
            header=None,
            names=['filename'])

        imgList = list(df['filename'].values)

        return imgList
