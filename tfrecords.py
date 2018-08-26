import os
import cv2
import numpy as np
import tensorflow as tf 
import facenet
# from PIL import Image


def read_and_decode(filename, img_height, img_width, flag):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    if flag == 'train' or flag == 'valid': 
        features = tf.parse_single_example(serialized_example,
                                        features={
                                            'label': tf.FixedLenFeature([], tf.int64),
                                            'img_raw' : tf.FixedLenFeature([], tf.string),
                                        })

        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img = tf.reshape(img, [img_height, img_width, 3])
        img = tf.cast(img, tf.float32)
        label = tf.cast(features['label'], tf.int32)
        return img, label
    else:
        features = tf.parse_single_example(serialized_example,
                                        features={
                                            'label': tf.FixedLenFeature([], tf.int64),
                                            'img1_raw' : tf.FixedLenFeature([], tf.string),
                                            'img2_raw' : tf.FixedLenFeature([], tf.string),
                                        })

        img1 = tf.decode_raw(features['img1_raw'], tf.uint8)
        img2 = tf.decode_raw(features['img2_raw'], tf.uint8)
        img1 = tf.reshape(img1, [img_height,img_width, 3])
        img2 = tf.reshape(img2, [img_height,img_width, 3])
        img1 = tf.cast(img1, tf.float32)
        img2 = tf.cast(img2, tf.float32)
        label = tf.cast(features['label'], tf.int32)
        return img1, img2, label

def stored_to_tfrecords(csv_file, tf_file, flag):
    line_num = 0
    label = 0
    writer = tf.python_io.TFRecordWriter(tf_file)
    with open(csv_file, "r") as f:
        for line in f.readlines():
            line_num += 1
            if line_num % 100 == 0:
                print('writing dataset records to line %d' % line_num)
            if flag == 'train' or flag == 'valid':
                path, label = line.strip().split()
                path = '../deepid1/' + path.replace('\\','/')
                img = cv2.imread(path)
                label = int(label)
                img_raw = img[:,:,::-1].tobytes()              #gbr->rgb,将图片转化为原生bytes
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
            else:
                if flag == 'test':
                    path1, path2, label = line.strip().split()
                    path1 = '../deepid1/' + path1.replace('\\','/')
                    path2 = '../deepid1/' + path2.replace('\\','/')
                else:
                    if line_num == 1:
                        str1, str2 = line.strip().split()
                        repeat_num = int(str1)
                        match_num = int(str2)
                        continue
                    if line_num % match_num == 2:
                        label = 1 - label
                    if label == 1:
                        str1, str2, str3 = line.strip().split()
                        path1 = '../../Datasets/lfw_mtcnnpy_160/{0}/{0}_{1:0>4}.png'.format(str1,str2)
                        path2 = '../../Datasets/lfw_mtcnnpy_160/{0}/{0}_{1:0>4}.png'.format(str1,str3)
                    else:
                        str1, str2, str3, str4 = line.strip().split()
                        path1 = '../../Datasets/lfw_mtcnnpy_160/{0}/{0}_{1:0>4}.png'.format(str1,str2)
                        path2 = '../../Datasets/lfw_mtcnnpy_160/{0}/{0}_{1:0>4}.png'.format(str3,str4)
                img1 = cv2.imread(path1)
                img2 = cv2.imread(path2)
                label = int(label)
                img1_raw = img1[:,:,::-1].tobytes()              #gbr->rgb,将图片转化为原生bytes
                img2_raw = img2[:,:,::-1].tobytes()              #gbr->rgb,将图片转化为原生bytes
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'img1_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img1_raw])),
                    'img2_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img2_raw]))
                }))
            writer.write(example.SerializeToString())  #序列化为字符串
    writer.close()


if __name__ == '__main__':
    # translate training data
    csv_file = './data/train_set.csv'
    tf_file = './data/train.tfrecords'
    # stored_to_tfrecords(csv_file, tf_file, 'train')

    # translate validating data
    csv_file = './data/valid_set.csv'
    tf_file = './data/valid.tfrecords'
    # stored_to_tfrecords(csv_file, tf_file, 'valid')

    # translate test data
    csv_file = './data/test_set.csv'
    tf_file = './data/test.tfrecords'
    # stored_to_tfrecords(csv_file, tf_file, 'test')

    # translate LFW dataset data
    txt_file = './data/pairs.txt'
    tf_file = './data/LFW_MTCNN.tfrecords'
    stored_to_tfrecords(txt_file, tf_file, 'LFW')

