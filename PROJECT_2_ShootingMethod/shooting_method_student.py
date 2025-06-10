#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：打靶法与scipy.solve_bvp求解边值问题 - 学生实现

本项目实现打靶法和scipy.solve_bvp两种方法来求解二阶线性常微分方程边值问题：
u''(x) = -π(u(x)+1)/4
边界条件：u(0) = 1, u(1) = 1

学生姓名：[段林焱 蔡宇航]
学号：[20231050098 20231050013]
完成日期：[2025/6/4]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_bvp
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')


def ode_system_shooting(y, t=None):
    """
    为打靶法定义ODE系统。

    将二阶ODE u'' = -π(u+1)/4 转化为一阶系统：
    y1 = u, y2 = u'
    y1' = y2
    y2' = -π(y1+1)/4

    参数:
        y (array or float): 状态向量 [y1, y2]，其中 y1=u, y2=u'，或时间 t
        t (float or array, optional): 自变量（时间/位置），或状态向量

    返回:
        list: 导数 [y1', y2']

    注：此函数可处理 (y, t) 和 (t, y) 两种参数顺序，以兼容不同求解器。
    """
    if isinstance(y, (int, float)) and hasattr(t, '__len__'):
        t, y = y, t  # 交换参数顺序
    return [y[1], -np.pi * (y[0] + 1) / 4]


def boundary_conditions_scipy(ya, yb):
    """
    为scipy.solve_bvp定义边界条件。

    边界条件：u(0) = 1, u(1) = 1
    ya[0] 应为 1, yb[0] 应为 1

    参数:
        ya (array): 左边界值 [u(0), u'(0)]
        yb (array): 右边界值 [u(1), u'(1)]

    返回:
        array: 边界条件残差
    """
    return np.array([ya[0] - 1, yb[0] - 1])


def ode_system_scipy(x, y):
    """
    为scipy.solve_bvp定义ODE系统。

    参数:
        x (float): 自变量
        y (array): 状态向量 [y1, y2]

    返回:
        array: 导数列向量
    """
    return np.vstack((y[1], -np.pi * (y[0] + 1) / 4))


def solve_bvp_shooting_method(x_span, boundary_conditions, n_points=100, max_iterations=10, tolerance=1e-6):
    """
    使用打靶法求解边值问题。

    算法：
    1. 猜测初始斜率 m1
    2. 以 [u(0), m1] 为初始条件求解 IVP
    3. 检查 u(1) 是否满足边界条件
    4. 若不满足，使用割线法调整斜率并重复

    参数:
        x_span (tuple): 求解区间 (x_start, x_end)
        boundary_conditions (tuple): 边界条件 (u_left, u_right)
        n_points (int): 离散点数量
        max_iterations (int): 最大迭代次数
        tolerance (float): 收敛容差

    返回:
        tuple: (x_array, y_array) 解的数组
    """
    x_start, x_end = x_span
    u_left, u_right = boundary_conditions
    x = np.linspace(x_start, x_end, n_points)

    # 初始斜率猜测
    m1 = -1.0
    y0 = [u_left, m1]
    sol1 = odeint(ode_system_shooting, y0, x)
    u_end_1 = sol1[-1, 0]

    if abs(u_end_1 - u_right) < tolerance:
        return x, sol1[:, 0]

    # 第二次猜测
    m2 = m1 * u_right / u_end_1 if abs(u_end_1) > 1e-12 else m1 + 1.0
    y0[1] = m2
    sol2 = odeint(ode_system_shooting, y0, x)
    u_end_2 = sol2[-1, 0]

    if abs(u_end_2 - u_right) < tolerance:
        return x, sol2[:, 0]

    # 割线法迭代
    for _ in range(max_iterations):
        if abs(u_end_2 - u_end_1) < 1e-12:
            m3 = m2 + 0.1
        else:
            m3 = m2 + (u_right - u_end_2) * (m2 - m1) / (u_end_2 - u_end_1)

        y0[1] = m3
        sol3 = odeint(ode_system_shooting, y0, x)
        u_end_3 = sol3[-1, 0]

        if abs(u_end_3 - u_right) < tolerance:
            return x, sol3[:, 0]

        m1, m2 = m2, m3
        u_end_1, u_end_2 = u_end_2, u_end_3

    print(f"警告：打靶法在 {max_iterations} 次迭代后未收敛。最终边界误差: {abs(u_end_3 - u_right):.2e}")
    return x, sol3[:, 0]


def solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points=50):
    """
    使用 scipy.solve_bvp 求解边值问题。

    参数:
        x_span (tuple): 求解区间 (x_start, x_end)
        boundary_conditions (tuple): 边界条件 (u_left, u_right)
        n_points (int): 初始网格点数

    返回:
        tuple: (x_array, y_array) 解的数组
    """
    x_start, x_end = x_span
    u_left, u_right = boundary_conditions
    x_init = np.linspace(x_start, x_end, n_points)

    # 初始猜测：线性插值
    y_init = np.zeros((2, x_init.size))
    y_init[0] = u_left + (u_right - u_left) * (x_init - x_start) / (x_end - x_start)
    y_init[1] = (u_right - u_left) / (x_end - x_start)

    sol = solve_bvp(ode_system_scipy, boundary_conditions_scipy, x_init, y_init)
    if not sol.success:
        raise RuntimeError(f"scipy.solve_bvp 求解失败: {sol.message}")

    x_fine = np.linspace(x_start, x_end, 100)
    y_fine = sol.sol(x_fine)[0]
    return x_fine, y_fine


def compare_methods_and_plot(x_span=(0, 1), boundary_conditions=(1, 1), n_points=100):
    """
    比较打靶法和 scipy.solve_bvp 的解，并生成对比图像。

    参数:
        x_span (tuple): 求解区间
        boundary_conditions (tuple): 边界条件
        n_points (int): 绘图点数

    返回:
        dict: 包含解和分析结果的字典
    """
    print("正在使用两种方法求解 BVP...")

    # 打靶法求解
    print("运行打靶法...")
    x_shoot, y_shoot = solve_bvp_shooting_method(x_span, boundary_conditions, n_points)

    # scipy.solve_bvp 求解
    print("运行 scipy.solve_bvp...")
    x_scipy, y_scipy = solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points // 2)

    # 插值比较
    y_scipy_interp = np.interp(x_shoot, x_scipy, y_scipy)
    max_diff = np.max(np.abs(y_shoot - y_scipy_interp))
    rms_diff = np.sqrt(np.mean((y_shoot - y_scipy_interp) ** 2))

    # 生成图像
    plt.figure(figsize=(12, 8))

    # 主对比图
    plt.subplot(2, 1, 1)
    plt.plot(x_shoot, y_shoot, 'b-', linewidth=2, label='打靶法')
    plt.plot(x_scipy, y_scipy, 'r--', linewidth=2, label='scipy.solve_bvp')
    plt.plot([x_span[0], x_span[1]], [boundary_conditions[0], boundary_conditions[1]],
             'ko', markersize=8, label='边界条件')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('BVP 求解方法对比')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 差异图
    plt.subplot(2, 1, 2)
    plt.plot(x_shoot, y_shoot - y_scipy_interp, 'g-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('差异 (打靶 - scipy)')
    plt.title(f'解的差异 (最大: {max_diff:.2e}, RMS: {rms_diff:.2e})')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 输出分析
    print("\n解的分析:")
    print(f"最大差异: {max_diff:.2e}")
    print(f"RMS 差异: {rms_diff:.2e}")
    print(f"打靶法边界: u(0) = {y_shoot[0]:.6f}, u(1) = {y_shoot[-1]:.6f}")
    print(f"scipy.solve_bvp 边界: u(0) = {y_scipy[0]:.6f}, u(1) = {y_scipy[-1]:.6f}")

    return {
        'x_shooting': x_shoot,
        'y_shooting': y_shoot,
        'x_scipy': x_scipy,
        'y_scipy': y_scipy,
        'max_difference': max_diff,
        'rms_difference': rms_diff
    }


if __name__ == "__main__":
    print("项目2：打靶法与scipy.solve_bvp求解边值问题")
    print("=" * 50)
    results = compare_methods_and_plot()
    print("项目完成！")
