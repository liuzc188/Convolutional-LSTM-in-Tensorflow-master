import os
import scipy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import utils_conv_lstm as u
from config import cfg

def_imagepath = 'D:\\0000139611_2\\SRAD\\SRAD2018_TRAIN_001'

def load_path(path = def_imagepath):
    p = os.listdir(path)
    SRADpath=[]
    for filename in p:
        filepath = os.path.join(path, filename)
        SRADpath.append(filepath)
    return SRADpath


def load_data( seq_length, shape, imagepath = def_imagepath, is_training = True):
    SRAD = load_path()
    imagepath = tf.cast(SRAD, tf.string)
    input_queue = tf.train.slice_input_producer([imagepath], shuffle=False)
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.resize_images(image, [shape, shape], method=0)
    image = tf.cast(image, tf.uint8)
    image_batch = tf.train.batch([image], batch_size=seq_length)
    dat = tf.reshape(image_batch,[1,seq_length,shape,shape,3])
    return dat

#在此处准备数据集
def generate_bouncing_ball_sample(batch_size, seq_length, shape, is_training):
  # for i in range(batch_size):
  #   dat[i, :, :, :, :] = load_data(seq_length, shape, is_training).eval()
  data_loader = load_data(seq_length, shape, is_training)
  image_batch = tf.train.batch([data_loader], batch_size=batch_size)
  dat = tf.reshape(image_batch, [batch_size, seq_length, shape, shape, 3])
  return dat

# 此处为实验程序
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     dd = load_data(30, 32)
#     dat = generate_bouncing_ball_sample(cfg.batch_size, cfg.seq_length, cfg.shape, cfg.is_training)
#     n=1

# def load_data(batch_size, is_training=True):
#     path = 'F:\\SRAD\\SRAD2018_TRAIN_001'
#     if is_training:
#         dat = np.zeros((batch_size, seq_length, shape, shape, 3)) #读入一个批矩阵
#
#         fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
#         loaded = np.fromfile(file=fd, dtype=np.uint8)
#         trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)
#
#         fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
#         loaded = np.fromfile(file=fd, dtype=np.uint8)
#         trainY = loaded[8:].reshape((60000)).astype(np.int32)
#
#         trX = trainX[:55000] / 255.
#         trY = trainY[:55000]
#
#         valX = trainX[55000:, ] / 255.
#         valY = trainY[55000:]
#
#         num_tr_batch = 55000 // batch_size
#         num_val_batch = 5000 // batch_size
#
#         return trX, trY, num_tr_batch, valX, valY, num_val_batch
#     else:
#         fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
#         loaded = np.fromfile(file=fd, dtype=np.uint8)
#         teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)
#
#         fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
#         loaded = np.fromfile(file=fd, dtype=np.uint8)
#         teY = loaded[8:].reshape((10000)).astype(np.int32)
#
#         num_te_batch = 10000 // batch_size
#         return teX / 255., teY, num_te_batch





def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs


# For version compatibility
def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


# For version compatibility
def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)
