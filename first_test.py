#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

# 构建一个方程 trY = 2 * trX，并且加入一些随机干扰项
trX = np.linspace(-1, 1, 101)
trY = 2 * trX + np.random.rand(*trX.shape) * 0.123

# 创建两个占位符，数据类型是 tf.float32
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 创建一个变量系数 w , 最后训练出来的值，应该接近 2
w = tf.Variable(0.0, name="weights")
y_model = tf.multiply(X, w)

# 定义损失函数 (Y - y_model)^2
cost = tf.square(Y - y_model)

# 定义学习率
learning_rate = 0.01

# 使用梯度下降来训练模型，学习率为 learning_rate , 训练目标是使损失函数最小
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
with tf.Session() as sess:
    # 初始化所有的变量
    init = tf.global_variables_initializer()
    sess.run(init)
    # 对模型训练100次
    for i in range(100):
        for (x, y) in zip(trX, trY):
            sess.run(train_op, feed_dict={X: x, Y: y})
    # 输出 w 的值
    print(sess.run(w))