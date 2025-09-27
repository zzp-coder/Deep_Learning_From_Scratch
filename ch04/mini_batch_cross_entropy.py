import numpy as np

from ch03.load_mnist import load_mnist

# x_train, t_train, x_test, t_test = load_mnist(flatten=True,one_hot_label=True)
# print(x_train.shape)
# print(t_train.shape)
# #
# #
# train_size = x_train.shape[0]
# batch_size = 10
# batch_mask = np.random.choice(train_size, batch_size)
# print(batch_mask)
# x_batch = x_train[batch_mask]
# t_batch = t_train[batch_mask]

# y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
# t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# t = [2]
# print(np.array(y).shape[0])

# def cross_entropy_error(y, t):
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
#     batch_size = y.shape[0]
#     return -np.sum(t * np.log(y+1e-7))/batch_size

# print(cross_entropy_error(np.array(y), np.array(t)))

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
# print(cross_entropy_error(np.array(y), np.array(t)))