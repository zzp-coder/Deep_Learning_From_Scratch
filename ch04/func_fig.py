import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义函数
def function_2(x):
    return x[0]**2 + x[1]**2

# 生成网格数据
x0 = np.linspace(-3, 3, 100)
x1 = np.linspace(-3, 3, 100)
X0, X1 = np.meshgrid(x0, x1)

# 计算函数值
Z = function_2([X0, X1])

# 创建三维图
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
ax.plot_surface(X0, X1, Z, cmap='viridis', edgecolor='none', alpha=0.8)

# 设置标签
ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('f(x0, x1)')
ax.set_title('f(x0, x1) = x0^2 + x1^2')

plt.show()