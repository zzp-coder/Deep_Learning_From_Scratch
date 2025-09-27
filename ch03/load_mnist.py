import numpy as np
import torchvision
from torchvision import transforms
def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    # 加载 MNIST 数据集
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                               transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                              transform=transforms.ToTensor())

    # 获取数据和标签
    x_train = train_dataset.data.numpy()  # 转为 NumPy 格式
    t_train = train_dataset.targets.numpy()
    x_test = test_dataset.data.numpy()
    t_test = test_dataset.targets.numpy()

    # 数据归一化
    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0

    # 展平数据
    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)  # (60000, 784)
        x_test = x_test.reshape(x_test.shape[0], -1)     # (10000, 784)

    # 标签转换为 one-hot 编码
    if one_hot_label:
        t_train = _to_one_hot(t_train, 10)
        t_test = _to_one_hot(t_test, 10)

    return x_train, t_train, x_test, t_test

def _to_one_hot(labels, num_classes):
    one_hot = np.zeros((labels.size, num_classes), dtype=np.float32)
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot