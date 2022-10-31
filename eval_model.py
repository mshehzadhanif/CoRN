""" Model Evaluation
    Competitive Residual Network (CoRN) for image classification
    Muhammad Shehzad Hanif (mshanif@kau.edu.sa)
    Muhammad Bilal (meftekar@kau.edu.sa)
    Last modified: 7/2/2018
"""

from __future__ import print_function

import os
import sys
import scipy.io
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.datasets import cifar100
from six.moves.urllib.request import urlopen
from preprocessing import preprocess

"""
    Main function to evaluate CoRN model on benchmark datasets    
"""
def main(argv):
    #parse arguments
    if(len(argv) < 3):
        print('Usage: python eval_model.py [dataset (cifar10/cifar100/svhn)] [model_name]')
        return
    dataset_name = argv[1]
    model_name = argv[2]
    print('dataset = %s, model_name = %s' %(dataset_name, model_name))

    #load dataset
    if dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        num_classes = 10
    elif dataset_name == 'cifar100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        num_classes = 100
    elif dataset_name == 'svhn':
        url = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
        filedata = urlopen(url)
        data = filedata.read()
        with open('test_32x32.mat', 'wb') as fd:
            fd.write(data)
        test_data = scipy.io.loadmat('test_32x32.mat')
        x_test = test_data['X'].transpose(3, 0, 1, 2)
        y_test = test_data['y'].reshape((-1))
        y_test[y_test == 10] = 0
        x_test = x_test.astype('float32')
        num_classes = 10
    else:
        print('Unknown dataset')
        return

    #preprocesing
    if dataset_name == 'svhn':
        # rescaling to [0,1]
        x_test = x_test / 255.0
    else:
        # mean/std preprocessing
        p = preprocess()
        p.fit(x_train)
        p.apply(x_test)

    #encoding of the labels
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    #load model
    model = load_model(model_name)

    #evaluate model
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    test_error = (1 - accuracy)*100
    print("%s test error = %1.2f%%" % (dataset_name, test_error))

if __name__ == '__main__':
    main(sys.argv)