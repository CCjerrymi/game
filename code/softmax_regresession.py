# softmax_regresession.py
# 运行时CPU有超频现象
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time
# 运行时系统提示CPU得不到利用，添加代码
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 获取输入数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# train.images.shap得出784，这意味着mnist.train.image里面保存着784个数字
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

sess = tf.InteractiveSession()
# 设置输出数据的地方，第一个参数表示参数的类型，第二个参数表示tensor的shape
# 也就是数据的尺寸，None表示无条件数的输入，784表示每一个数是784维度的向量
x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 实现公式y = softmax(Wx + b)
# Wx + b的值为证明图片是某个数字的证明，b为偏置量，因为图片中存在各种各样的干扰
# y则是softmax将证据转化为概率
y = tf.nn.softmax(tf.matmul(x, w) + b)
# 定义损失函数
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# 定义优化函数,调用tf.train.GradientDescentOptimizer，设置学习率为0.5，优化目标为cross_entry
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 使用tensorflow的全局参数初始化器tf.global_variables_initializer()，直接执行run方法
tf.global_variables_initializer().run()
# 迭代进行训练操作

# 标记时间
start_time = time.time()
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

end_time = time.time()
print("迭代训练使用了：", (end_time - start_time))
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 统计全部样本预测的accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 打印结果
print(accuracy.eval({x: mnist.test.images, y_:mnist.test.labels}))


# 将数据转化成可视化的图像
# data_trainsform这个函数是将数字转化成可视化的图像
def data_trainsform(a):  # 将784转换成28*28的矩阵
    b = np.zeros([28, 28])  # 定义一个简单的28X28矩阵
    for i in range(0, 27):
        for j in range(0, 27):
            b[i][j] = a[28 * i + j]
    return b


tile = data_trainsform(mnist.train.images[1])
print(mnist.train.labels[1])
plt.figure()
plt.imshow(tile)
plt.show()
print(mnist.train.images.shape)
print(mnist.train.labels.shape)
print(mnist.validation.images.shape)
print(mnist.validation.labels.shape)
print(mnist.test.images.shape)
print(mnist.test.labels.shape)

