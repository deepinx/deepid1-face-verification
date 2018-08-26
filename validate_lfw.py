'''
预测 人脸验证集
'''

# ! /usr/bin/python
import dlib
import cv2
import facenet
import numpy as np
import tensorflow as tf
from tfrecords import read_and_decode
import matplotlib.pyplot as plt
# from vec import read_csv_pair_file

# print('reading test dataset from data/testset.pkl ...')
# with open('data/testset.pkl', 'rb') as f:
#     testX1 = pickle.load(f)
#     testX2 = pickle.load(f)
#     testY  = pickle.load(f)

# 计算人脸的128维特征向量
def create128DVectorSpace(img_batch):
    predicter_path = '../face_alignment/model/shape_predictor_5_face_landmarks.dat'
    face_rec_model_path = '../face_alignment/model/dlib_face_recognition_resnet_model_v1.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predicter_path)
    face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

    face_descriptor = np.ones(128)
    face_descriptors = []
    for n in range(img_batch.shape[0]):
        rgb_img = img_batch[n].astype(np.uint8)

        dets = detector(rgb_img, 1)
        print("Image {} --> Number of faces detected: {}".format(n+1, len(dets)))
        for index, det in enumerate(dets):
            shape = predictor(rgb_img, det)   # 提取68个特征点
            face_descriptor = face_rec_model.compute_face_descriptor(rgb_img, shape)    # 计算人脸的128维特征向量
        face_descriptors.append(face_descriptor)
    
    return face_descriptors

def calc_facenet_features(img_batch, sess):
    image_size = 160 #don't need equal to real image size, but this value should not small than this
    modeldir = '../understand_facenet/models/20170512-110547.pb' #change to your model dir

    facenet.load_model(modeldir)
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    image_nums = img_batch.shape[0]
    embedding_size = embeddings.get_shape()[1]
    emb_array = np.zeros((image_nums, embedding_size))
    for n in range(image_nums):
        rgb_img = img_batch[n]
        # rgb_img = cv2.resize(rgb_img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        rgb_img = facenet.prewhiten(rgb_img)
        img_reshape = rgb_img.reshape(-1,image_size,image_size,3)
        feed_dict = { images_placeholder:img_reshape, phase_train_placeholder:False }
        emb_array[n, :] = sess.run(embeddings, feed_dict=feed_dict)
        print("Calculating features of Image {} ... ".format(n+1))
        
    return emb_array

# 求余弦距离阈值
def part_mean(x, mask):
    z = x * mask
    # 一致组余弦距离总和/一致组数量
    return float(np.sum(z) / np.count_nonzero(z))

if __name__ == '__main__':

    print('Reading test dataset from data/LFW.tfrecords ...')
    # img1, img2, label = read_and_decode("data/LFW_55_47.tfrecords", 55, 47, 'test')
    img1, img2, label = read_and_decode("data/LFW_MTCNN_160.tfrecords", 160, 160, 'test')
    test_batch_x1, test_batch_x2, test_batch_y = tf.train.batch([img1, img2, label], 600, capacity=2000)

    with tf.Session() as sess:

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # from deepid1 import h0, h5, saver
        # saver.restore(sess, 'checkpoint/30000.ckpt')
        testX1, testX2, testY = sess.run([test_batch_x1, test_batch_x2, test_batch_y])

        # h1 = sess.run(h5, {h0: testX1})
        # h2 = sess.run(h5, {h0: testX2})

        # h1 = create128DVectorSpace(testX1)
        # h2 = create128DVectorSpace(testX2)

        h1 = calc_facenet_features(testX1, sess)
        h2 = calc_facenet_features(testX2, sess)

        np.savetxt('./data/h1.txt', h1, fmt='%.7f')
        np.savetxt('./data/h2.txt', h2, fmt='%.7f')

        # h1 = np.loadtxt('./data/h1_MTCNN.txt')
        # h2 = np.loadtxt('./data/h2_MTCNN.txt')

        coord.request_stop()
        coord.join(threads)

    from scipy.spatial.distance import cosine, euclidean
    # 预测结果（余弦相似度（集）） consine实际上是1-余弦值
    # pre_y = np.array([cosine(x, y) for x, y in zip(h1, h2)])
    # 预测结果（欧式距离） Euclidean distance
    pre_y = np.array([euclidean(x, y) for x, y in zip(h1, h2)])

    true_mean = part_mean(pre_y, testY)  # 一致余弦距离均值
    false_mean = part_mean(pre_y, 1 - testY)  # 非一致余弦距离均值
    # best_threshold = (true_mean + false_mean) / 2
    thresholds = np.arange(0, 2, 0.01)
    accuracy = np.zeros((len(thresholds)))
    for index, threshold in enumerate(thresholds):
        accuracy[index] = np.mean((pre_y < threshold) == testY.astype(bool))
    best_threshold_index = np.argmax(accuracy)
    best_threshold = thresholds[best_threshold_index]
    test_accuracy = accuracy[best_threshold_index]
    print('true mean %g' % true_mean)
    print('false mean %g' % false_mean)
    print('best threshold %g' % best_threshold)
    print('test accuracy %g' % test_accuracy)

    pre_y_true = []
    pre_y_false = []
    for i in range(len(testY)):
        if testY[i] == 1:
            pre_y_true.append(pre_y[i])
        else:
            pre_y_false.append(pre_y[i])

    plt.hist(pre_y_true, 50, density=1, facecolor='g', alpha=0.75, histtype='step')
    plt.hist(pre_y_false, 50, density=1, facecolor='r', alpha=0.75, histtype='step')
    plt.show()
