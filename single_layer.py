import input_data
import tensorflow as tf

"""
单层神经网络 
参考 https://github.com/jorditorresBCN/FirstContactWithTensorFlow-2ndEdition

"""

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# tensor x用来存储MNIST图像向量中784个浮点值
# None代表该维度可为任意大小，本例子中将会是学习过程中照片的数量
x = tf.placeholder("float", [None, 784])

# 创建两个变量来保存权重W和偏置b，本例中用常量0来初始化tensor
# W 的shape为[Dimension(784), Dimension(10)],这是由它的参数常量tensor tf.zeros[784,10]定义的
# 参数b也是类似，由它的参数指定形状为[Dimension(10)]
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 实现模型。将图像向量x与权重矩阵W相乘再加上b之后的结果作为参数传给softmax函数
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 创建一个placeholder代表正确的标签
y_ = tf.placeholder("float", [None, 10])

# 计算交叉熵作为代价函数（cost function）
# 使用tensorflow中内置的tf.log()对每一个元素y求对数，然后再与每一个y_的元素相乘
# 最后使用tf.reduce_sum对tensor的所有元素求和
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 使用反向传播算法与梯度下降算法来最小化交叉熵，同时学习速率为0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# tf.argmax(y,1)为模型中输入数据的最大概率标签，tf.argmax(y_,1)是实际的标签
# tf.equal方法比较预测结果与实际结果是否相等，返回值是一个布尔列表
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 布尔值转换成浮点数，例如，[True, False, True, True]会转换成[1,0,1,1]，其平均值0.75代表了准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 创建tf.Session()开始计算各操作
sess = tf.Session()
# 初始化所有变量
sess.run(tf.global_variables_initializer())

# 重复执行train_step训练模型，在此迭代1000次
for i in range(1000):
    # 每次迭代中，从训练数据集中随机选取100张图片作为一批
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 获得的输入分别赋给相关的placeholders
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    if i % 10 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
        print("Step %d - Train accuracy %.3f" % (i, train_accuracy))

# 使用mnist.test数据集作为feed_dict参数来计算准确率
test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print("Test accuracy %.3f" % test_accuracy)