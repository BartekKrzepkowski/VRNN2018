import tensorflow
import numpy as np
from keras import datasets
from tensorflow.examples.tutorials.mnist import input_data




def load_mnist_data():
    mnist = datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    return (x_train, y_train),(x_test, y_test)


def load_fashion_mnist_data():
    data = input_data.read_data_sets('data/fashion')
    BATCH_SIZE = 1000000
    x_train, y_train = data.train.next_batch(BATCH_SIZE)
    x_test, y_test = data.test.next_batch(BATCH_SIZE)

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return (x_train, y_train),(x_test, y_test)


def load_pets_data():
    raw_data = {"fname": [], "label": []}
    for fname in os.listdir("datasets/dogs_vs_cats/train"):
        raw_data['fname'].append(fname)
        raw_data['label'].append(fname[:3])
        
    
        
        
    return (x_train, y_train),(x_test, y_test)

def load_cifar10_data():
    cifar10 = datasets.cifar10
    (x_train, y_train),(x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return (x_train, y_train),(x_test, y_test)


def load_cifar100_data():
    cifar100 = datasets.cifar100
    (x_train, y_train),(x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return (x_train, y_train),(x_test, y_test)
    

