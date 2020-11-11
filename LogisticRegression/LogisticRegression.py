# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = train_set_y.shape[1]  # 训练集里图片的数量。
m_test = test_set_y.shape[1]  # 测试集里图片的数量。
num_px = train_set_x_orig.shape[1]  # 训练、测试集里面的图片的宽度和高度（均为64x64）。

# 现在看一看我们加载的东西的具体情况
print("训练集的数量: m_train = " + str(m_train))
print("测试集的数量 : m_test = " + str(m_test))
print("每张图片的宽/高 : num_px = " + str(num_px))
print("每张图片的大小 : (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("训练集_图片的维数 : " + str(train_set_x_orig.shape))
print("训练集_标签的维数 : " + str(train_set_y.shape))
print("测试集_图片的维数: " + str(test_set_x_orig.shape))
print("测试集_标签的维数: " + str(test_set_y.shape))

# 将训练集的维度降低并转置。就是把所有图片全部放到一个矩阵中
# reshape()在不改变数组或者说矩阵值的情况下改变其行列大小 -1表示不知道具体行列的值为多少
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
# 将测试集的维度降低并转置。同上
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print("训练集降维最后的维度： " + str(train_set_x_flatten.shape))
print("训练集_标签的维数 : " + str(train_set_y.shape))
print("测试集降维之后的维度: " + str(test_set_x_flatten.shape))
print("测试集_标签的维数 : " + str(test_set_y.shape))
# 标准化数据让数据都位于[0,1]之间
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s
def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0
    # 使用断言来确保我要的数据是正确的
    assert (w.shape == (dim, 1))  # w的维度是(dim,1)
    assert (isinstance(b, float) or isinstance(b, int))  # b的类型是float或者是int
    return w, b
def propagate(w, b, X, Y):
    m = X.shape[1]
    # 正向传播
    A = sigmoid(np.dot(w.T, X) + b)  # 预测值矩阵
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # 计算代价矩阵
    # 反向传播（使用导数定义计算的偏导数）
    dw = (1 / m) * np.dot(X, (A - Y).T)  # 请参考视频中的偏导公式。
    db = (1 / m) * np.sum(A - Y)  # 请参考视频中的偏导公式。
    # 使用断言确保我的数据是正确的
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    # 创建一个字典，把dw和db保存起来。
    grads = {
        "dw": dw,
        "db": db
    }
    return grads, cost
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):
        # 传播
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]
        # 梯度下降
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 记录成本
        if i % 100 == 0:
            costs.append(cost)
        # 打印成本数据
        print("迭代的次数: %i ， 误差值： %f" % (i, cost))

    params = {
        "w": w,
        "b": b}
    grads = {
        "dw": dw,
        "db": db}
    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]  # 图片的数量
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # 计预测猫在图片中出现的概率
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        # 将概率a [0，i]转换为实际预测p [0，i]
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    # 使用断言
    assert (Y_prediction.shape == (1, m))
    return Y_prediction
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.05, print_cost=False):
    # X_train训练集 12288*209
    # Y_train 标签 1*209
    # 生成初始w,b向量维度为训练集的列长度
    w, b = initialize_with_zeros(X_train.shape[0])
    # 梯度下降
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # 从字典“参数”中检索参数w和b
    w, b = parameters["w"], parameters["b"]

    # 预测测试/训练集的例子
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # 打印训练后的准确性
    print("训练集准确性：", format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100), "%")
    print("测试集准确性：", format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100), "%")

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediciton_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations}
    return d


d = model(train_set_x, train_set_y, test_set_x, test_set_y,2000, 0.005,True)

# 绘制图
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()
