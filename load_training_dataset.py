
from __future__ import print_function
import os
import glob
import cv2
from sklearn.utils import shuffle
import numpy as np

def load_train(train_path, classes, num_channels):
    '''
    Function:
    This function loads the images in the training directory.
    Arguments:
        train_path : Path of the training dataset
        classes    : The classes on which the model is trained to classify
    Return:
        images     : Numpy array of images in the training dataset dircetory.
        labels     : Numpy array of labels in the training dataset dircetory.
        img_names  : Image names.
        cls        : Numpy array of class to which the image belongs to.
    '''
    images = list()
    labels = list()
    img_names = list()
    cls = list()
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Going to read training images')
    for fields in classes:
        index = classes.index(fields)
        logger.info('Now going to read {} files (Index: {})'.format(fields, index))
        path1 = os.path.join(train_path, fields, '*g')
        path2 = os.path.join(train_path, fields, '*G')
        files = glob.glob(path1) + glob.glob(path2)

        # Reading all the images in the training dataset directory
        for imagename in files:
            if num_channels == 3:
                input_image = cv2.imread(imagename)

                # To convert BGR image to RGB
                image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            elif num_channels == 1:
                image = cv2.imread(imagename, 0)

            # Converting the images to numpy array so that the created numpy
            # arrays can be fed into the placeholders while training
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)

            # Appending the converted numpy arrays to list
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(imagename)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls


class DataSet(object):
    '''
    Dataset class.
    This class contains initialization function.
    '''
    def __init__(self, images, labels, img_names, cls):
        self._num_examples = images.shape[0]

        self._images = images
        self._labels = labels
        self._img_names = img_names
        self._cls = cls
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def img_names(self):
        return self._img_names

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # After each epoch we update this
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            # assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._img_names[start:end],\
            self._cls[start:end]


def read_train_sets(train_path, classes, num_channels, validation_size):
    '''
    Function:
    This function reads the images in the training dataset directory.
    Arguments:
        train_path      : Path of the training dataset
        classes         : The classes on which the model is trained to classify.
        validation_size : The ratio by which the images should be taken for training
        and validation.
    Return:
        data_sets       : Images for training and validation.
    '''
    class DataSets(object):
        '''
        Datasets class.
        '''
        pass
    data_sets = DataSets()

    # Load_train function returns the vectors for the images
    # in the training dataset directory with the corresponding labels
    images, labels, img_names, cls = load_train(train_path, classes, num_channels)

    # Shuffling all the images in the training dataset directory
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls)

    # Calculating the validation size from the batch size
    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    # Validation images
    validation_images = images[:validation_size]

    # Validation labels
    validation_labels = labels[:validation_size]

    # Validation image names
    validation_img_names = img_names[:validation_size]

    # Validation classes
    validation_cls = cls[:validation_size]

    # Training images
    train_images = images[validation_size:]

    # Training labels
    train_labels = labels[validation_size:]

    # Training image names
    train_img_names = img_names[validation_size:]

    # Training classes
    train_cls = cls[validation_size:]

    data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
    data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, \
        validation_cls)
    return data_sets
