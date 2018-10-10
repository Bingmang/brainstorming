import tensorflow as tf


class ImageDataSet:
    """
    目录下必须有train/validation/test目录
    """

    def __init__(self, image_path, labeled=False, label_path=None):

        self.image_path = image_path
        self.labeled = labeled
        self.label_path = label_path

        self._images = None
        self._labels = None

    def next_batch(self, batch_size=100, shuffle=True):
        pass

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels
