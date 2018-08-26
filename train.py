#! /usr/bin/python
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage import io

from deepid1 import *
from tfrecords import read_and_decode
from vec import read_csv_file


# testX1, testX2, testY, validX, validY, trainX, trainY = load_data()
# class_num = np.max(trainY) + 1

def save_batch_images(path, image_batch, start, stop):
    if not os.path.exists(path):
        os.mkdir(path)
    for n in range(start, stop+1):
        if n <= image_batch.shape[0]:
            # io.imsave(path+'/out%d.jpg' % n, np.uint8(image_batch[n]))
            # io.imshow(np.uint8(image_batch[n]))
            # plt.show()
            # img = Image.fromarray(np.uint8(image_batch[n]))
            # img.save(path+'/out%d.jpg' % n)
            # img.show()
            min = np.min(image_batch[n])
            max = np.max(image_batch[n])
            cv2.imshow('image', (image_batch[n][:,:,::-1]-min)/(max-min))    # rgb -> bgr
            cv2.waitKey(20)
            cv2.imwrite(path+'/out%d.jpg' % n, image_batch[n][:,:,::-1])
            # np.set_printoptions(threshold=np.inf)
            # print(image_batch[n])
            np.savetxt(path+'/data%d.txt' % n, image_batch[n][:,:,0], fmt='%.2f')
            print('saving image out%d.jpg' % n)
    cv2.destroyWindow('image')


if __name__ == '__main__':

    # data_x = trainX
    # data_y = (np.arange(class_num) == trainY[:,None]).astype(np.float32)
    # print('reading data from ../deepid1/data/valid_set.csv ...')
    # validX, validY = read_csv_file('../deepid1/data/valid_set.csv')
    # validY = (np.arange(class_num) == validY[:,None]).astype(np.float32)
    # with open('data/validset.pkl', 'rb') as f:
    #     validX = pickle.load(f)
    #     validY = pickle.load(f)
    
    img, label = read_and_decode("data/train.tfrecords",img_height, img_width, 'train')
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.3)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    label = tf.one_hot(label, class_num)
    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=64, capacity=2000, min_after_dequeue=200)
    img, label = read_and_decode("data/valid.tfrecords",img_height, img_width, 'valid')
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.3)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    label = tf.one_hot(label, class_num)
    validX, validY = tf.train.batch([img, label], batch_size=6415, capacity=2000)
    
    logdir = 'log'
    if tf.gfile.Exists(logdir):
        tf.gfile.DeleteRecursively(logdir)
    tf.gfile.MakeDirs(logdir)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(logdir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(logdir + '/test', sess.graph)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver.restore(sess, 'checkpoint/30000.ckpt')
        validX, validY = sess.run([validX, validY])
        # save_batch_images('./save', validX, 0, 63)    # for debug
        valid_accuracy = sess.run(accuracy, feed_dict={h0: validX, y_: validY})
        print('------->validating accuracy %g' % valid_accuracy)
        
        import time
        start_time = time.time()
        for i in range(30001,40001):
            batch_x, batch_y= sess.run([img_batch, label_batch])
            sess.run(train_step, feed_dict={h0: batch_x, y_: batch_y})
            if i % 100 == 0:
                summary, train_accuracy = sess.run([merged, accuracy], feed_dict={h0: batch_x, y_: batch_y})
                train_writer.add_summary(summary, i)
                print('step %d, training accuracy %g' % (i, train_accuracy)) 
            if i % 1000 == 0 and i != 0:
                end_time = time.time()
                print('1000 steps time: %ds' % (end_time-start_time))
                start_time = end_time
                summary, valid_accuracy = sess.run([merged, accuracy], feed_dict={h0: validX, y_: validY})
                test_writer.add_summary(summary, i)
                print('------->validating accuracy %g' % valid_accuracy)
                saver.save(sess, 'checkpoint/%05d.ckpt' % i)
                print('save model to checkpoint/%05d.ckpt' % i)

        coord.request_stop()
        coord.join(threads)
