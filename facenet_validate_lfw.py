'''
预测 人脸验证集
'''

# ! /usr/bin/python
import dlib
import cv2
import facenet
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def get_files(data_file):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    line_num = 0
    label = 0
    image_list = []
    label_list = []
    with open(data_file, "r") as f:
        for line in f.readlines():
            line_num += 1
            if line_num == 1:
                str1, str2 = line.strip().split()
                repeat_num = int(str1)
                match_num = int(str2)
                continue
            if line_num % match_num == 2:
                label = 1 - label
            if label == 1:
                str1, str2, str3 = line.strip().split()
                path1 = '{0}/{1}/{1}_{2:0>4}.{3}'.format(dataset_path,str1,str2,image_type)
                path2 = '{0}/{1}/{1}_{2:0>4}.{3}'.format(dataset_path,str1,str3,image_type)
            else:
                str1, str2, str3, str4 = line.strip().split()
                path1 = '{0}/{1}/{1}_{2:0>4}.{3}'.format(dataset_path,str1,str2,image_type)
                path2 = '{0}/{1}/{1}_{2:0>4}.{3}'.format(dataset_path,str3,str4,image_type)

            image_list.append(path1)
            image_list.append(path2)
            label_list.append(int(label))
            label_list.append(int(label))

    print('There are %d images in the datasets' %(len(image_list)))
    
    return image_list, label_list

def get_batch(img_list, label_list, img_height, img_width, batch_size, num_threads=1, capacity=2000):

    image = tf.cast(img_list, tf.string)
    label = tf.cast(label_list, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label], shuffle=False)  #shuffle=False，否则不会按顺序进行读取

    label = input_queue[1]

    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_png(image_contents, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, img_height, img_width)
    image = tf.image.per_image_standardization(image) # 将图片标准化

    image_batch, label_batch = tf.train.batch([image, label], batch_size, num_threads, capacity)    #num_threads=1，否则不会按顺序进行读取
    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch


# 求余弦距离阈值
def part_mean(x, mask):
    z = x * mask
    # 一致组余弦距离总和/一致组数量
    return float(np.sum(z) / np.count_nonzero(z))

if __name__ == '__main__':
    dataset_path = '../../Datasets/lfw_mtcnnpy_160'
    image_type = 'png'
    batch_size = 100
    image_size = 160
    modeldir = '../understand_facenet/models/20170512-110547.pb' #change to your model dir

    facenet.load_model(modeldir)
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    print('Reading test data from {}' .format(dataset_path))
    test_x, test_y = get_files("./data/pairs.txt")
    test_batch_x, test_batch_y = get_batch(test_x, test_y, image_size, image_size, batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())   # tf.train.string_input_producer need this initializer

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        nrof_images = int(len(test_x))
        nrof_batches = int(nrof_images/batch_size)
        embedding_size = embeddings.get_shape()[1]
        emb_array = np.zeros((nrof_images, embedding_size))
        label = np.zeros((nrof_images))
        for i in range(nrof_batches):
            start_index = i*batch_size
            end_index = min((i+1)*batch_size, nrof_images)
            testX, testY = sess.run([test_batch_x, test_batch_y])
            
            feed_dict = { images_placeholder:testX, phase_train_placeholder:False }
            emb_array[start_index:end_index] = sess.run(embeddings, feed_dict=feed_dict)
            label[start_index:end_index] = testY
            print('Processing step {} of {} ...' .format(i+1, nrof_batches))

        h1 = emb_array[0::2]
        h2 = emb_array[1::2]
        label = label[0::2]

        # np.savetxt('./data/h1_facenet.txt', h1, fmt='%.7f')
        # np.savetxt('./data/h2_facenet.txt', h2, fmt='%.7f')
        # np.savetxt('./data/label_facenet.txt', label, fmt='%.7f')

        # h1 = np.loadtxt('./data/h1_MTCNN.txt')
        # h2 = np.loadtxt('./data/h2_MTCNN.txt')

        coord.request_stop()
        coord.join(threads)

    from scipy.spatial.distance import euclidean
    # 预测结果（欧式距离） Euclidean distance
    pre_y = np.array([euclidean(x, y) for x, y in zip(h1, h2)])

    true_mean = part_mean(pre_y, label)  # 一致余弦距离均值
    false_mean = part_mean(pre_y, 1 - label)  # 非一致余弦距离均值
    # best_threshold = (true_mean + false_mean) / 2
    thresholds = np.arange(0, 2, 0.01)
    accuracy = np.zeros((len(thresholds)))
    for index, threshold in enumerate(thresholds):
        accuracy[index] = np.mean((pre_y < threshold) == label.astype(bool))
    best_threshold_index = np.argmax(accuracy)
    best_threshold = thresholds[best_threshold_index]
    test_accuracy = accuracy[best_threshold_index]
    print('true mean %g' % true_mean)
    print('false mean %g' % false_mean)
    print('best threshold %g' % best_threshold)
    print('test accuracy %g' % test_accuracy)

    pre_y_true = []
    pre_y_false = []
    for i in range(len(label)):
        if label[i] == 1:
            pre_y_true.append(pre_y[i])
        else:
            pre_y_false.append(pre_y[i])

    plt.hist(pre_y_true, 50, density=1, facecolor='g', alpha=0.75, histtype='step')
    plt.hist(pre_y_false, 50, density=1, facecolor='r', alpha=0.75, histtype='step')
    plt.show()
