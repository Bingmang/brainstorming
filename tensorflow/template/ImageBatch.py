import os

import tensorflow as tf
from PIL import Image


class ImageBatch:
    """
    自动生成tf.train.batch
    """

    def __init__(self, image_dir, image_size, batch_size):
        self._image_dir = image_dir
        self._image_size = image_size
        self._batch_size = batch_size
        self._image_path = []
        self._labels = []

    def __getPathAndLabel(self):
        pass

    def getBatch(self):
        pass
