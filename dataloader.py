import tensorflow as tf
import scipy.io as sio    
import numpy as np
import glob

def Dataloader(name, home_path, model_name):
    if name == 'cifar10':
        return Cifar10(home_path, model_name)
    
def Cifar10(home_path, model_name):
    from tensorflow.keras.datasets.cifar10 import load_data
    (train_images, train_labels), (val_images, val_labels) = load_data()
    
    @tf.function
    def pre_processing(image, is_training):
        with tf.name_scope('preprocessing'):
            image = tf.cast(image, tf.float32)
            image = (image-128)/128
            if is_training:
                image = tf.image.random_flip_left_right(image)
                sz = tf.shape(image)
                image = tf.pad(image, [[0,0],[4,4],[4,4],[0,0]], 'REFLECT')
                image = tf.image.random_crop(image,sz)
        return image
    
    return train_images, train_labels, val_images, val_labels, pre_processing

