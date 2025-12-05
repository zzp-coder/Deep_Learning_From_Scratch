import numpy as np
import matplotlib.pyplot as plt

# 定义函数和梯度
def f(x, y):
    return (1/5) * x**2 + y**2

def grad(x, y):
    df_dx = (1/10) * x   # 对 x 的偏导
    df_dy = 2 * y        # 对 y 的偏导
    return df_dx, df_dy

# 定义坐标网格
x = np.linspace(-10, 10, 20)
y = np.linspace(-10, 10, 20)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# 计算梯度
U, V = grad(X, Y)

# ---------- 画三维曲面 ----------
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_title("Surface of f(x,y) = 1/20*x^2 + y^2")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("f(x,y)")

# ---------- 画梯度场 ----------
ax2 = fig.add_subplot(1, 2, 2)
ax2.contour(X, Y, Z, levels=20, cmap="viridis")  # 等高线
ax2.quiver(X, Y, -U, -V, color="red")  # 梯度箭头
ax2.set_title("Gradient Field of f(x,y)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_aspect("equal")

plt.tight_layout()
plt.show()