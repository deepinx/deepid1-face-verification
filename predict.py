'''
预测 人脸验证集
'''

# ! /usr/bin/python
from deepid1 import *
import tensorflow as tf
from scipy.spatial.distance import cosine
from tfrecords import read_and_decode
import matplotlib.pyplot as plt
# from vec import read_csv_pair_file

# print('reading test dataset from data/testset.pkl ...')
# with open('data/testset.pkl', 'rb') as f:
#     testX1 = pickle.load(f)
#     testX2 = pickle.load(f)
#     testY  = pickle.load(f)


if __name__ == '__main__':
    print('reading test dataset from data/test.tfrecords ...')
    img1, img2, label = read_and_decode("data/test.tfrecords", 'test')
    # img1 = tf.image.random_flip_left_right(img1)
    # img1 = tf.image.random_brightness(img1, max_delta=0.3)
    # img1 = tf.image.random_contrast(img1, lower=0.8, upper=1.2)
    testX1, testX2, testY = tf.train.batch([img1, img2, label], batch_size=3120, capacity=2000)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver.restore(sess, 'checkpoint/30000.ckpt')
        testX1, testX2, testY = sess.run([testX1, testX2, testY])
        h1 = sess.run(h5, {h0: testX1})
        h2 = sess.run(h5, {h0: testX2})

        coord.request_stop()
        coord.join(threads)

    # 预测结果（余弦相似度（集）） consine实际上是1-余弦值
    pre_y = np.array([cosine(x, y) for x, y in zip(h1, h2)])

    # 求余弦距离阈值
    def part_mean(x, mask):
        z = x * mask
        # 一致组余弦距离总和/一致组数量
        return float(np.sum(z) / np.count_nonzero(z))


    true_mean = part_mean(pre_y, testY)  # 一致余弦距离均值
    false_mean = part_mean(pre_y, 1 - testY)  # 非一致余弦距离均值
    test_accuracy = np.mean((pre_y < (true_mean + false_mean) / 2) == testY.astype(bool))
    print('true mean %g' % true_mean)
    print('false mean %g' % false_mean)
    print('test accuracy %g' % test_accuracy)

    # pre_y_true = []
    # pre_y_false = []
    # for i in range(len(testY)):
    #     if testY[i] == 1:
    #         pre_y_true.append(pre_y[i])
    #     else:
    #         pre_y_false.append(pre_y[i])

    # plt.hist(pre_y_true, 50, density=1, facecolor='g', alpha=0.75, histtype='step')
    # plt.hist(pre_y_false, 50, density=1, facecolor='r', alpha=0.75, histtype='step')
    # plt.show()
