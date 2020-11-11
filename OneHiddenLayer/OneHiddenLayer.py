# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset
from testCases import *

np.random.seed(1)  # 设置一个固定的随机种子，以保证接下来的步骤中结果是一致的。
X, Y = load_planar_dataset()  # X存坐标，Y存真实值 一列为一个样本
shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1]  # 训练集里面的数量
print("X的维度为: " + str(shape_X))
print("Y的维度为: " + str(shape_Y))
print("数据集里面的数据有：" + str(m) + " 个")


def layer_sizes(X, Y):  # 神经网络层数设置
    n_x = X.shape[0]  # 输入层
    n_h = 4  # 隐藏层节点数
    n_y = Y.shape[0]  # 输出层
    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):  # 初始化随机W，B矩阵
    np.random.seed(2)  # 指定一个随机种子，以便你的输出与我们的一样。
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))
    # 使用断言确保我的数据格式是正确的
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    # 前向传播计算A2
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)  # 第一层400个样本的预测值
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)  # 输出层400个样本的预测值
    # 使用断言确保我的数据格式是正确的
    assert (A2.shape == (1, X.shape[1]))
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return A2, cache


def compute_cost(A2, Y, parameters):
    m = Y.shape[1]
    # 计算成本 multiply（a,b）a,b矩阵对应元素相乘（不是矩阵乘法）
    cost = - np.sum(np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))) / m
    assert (isinstance(cost, float))  # isinstance(a,b)判断a,b类型是否相同
    return cost


def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    dZ2 = A2 - Y  # 真实导数为A2，减去真实值来判断方向
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    W1, W2 = parameters["W1"], parameters["W2"]
    b1, b2 = parameters["b1"], parameters["b2"]
    dW1, dW2 = grads["dW1"], grads["dW2"]
    db1, db2 = grads["db1"], grads["db2"]
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


def nn_model(X, Y, n_h, num_iterations, print_cost=False):
    np.random.seed(3)  # 指定随机种子
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)  # 向前传播
        cost = compute_cost(A2, Y, parameters)  # 计算代价函数
        grads = backward_propagation(parameters, cache, X, Y)  # 向后传播
        parameters = update_parameters(parameters, grads, learning_rate=0.5)  # 梯度下降
        if print_cost:
            if i % 1000 == 0:
                print("第 ", i, " 次循环，成本为：" + str(cost))
    return parameters


def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)  # round函数四舍五入
    return predictions


parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)

# 绘制边界
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
predictions = predict(parameters, X)
print('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
plt.show()

hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]  # 隐藏层数量
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i + 1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations=5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    plt.show()
    print("隐藏层的节点数量： {}  ，准确率: {} %".format(n_h, accuracy))
