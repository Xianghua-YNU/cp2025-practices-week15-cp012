#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：打靶法与scipy.solve_bvp求解边值问题

本项目要求实现打靶法和scipy.solve_bvp两种方法来求解二阶线性常微分方程边值问题：
u''(x) = -π(u(x)+1)/4
边界条件：u(0) = 1, u(1) = 1
学生姓名：[]
学号：[YOUR_STUDENT_ID]
完成日期：[COMPLETION_DATE]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import fsolve
import warnings
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
warnings.filterwarnings('ignore')


def ode_system_shooting(t, y):
    """
    为打靶法定义ODE系统

    将二阶ODE u'' = -π(u+1)/4转化为一阶系统:
    y1 = u, y2 = u'
    y1' = y2
    y2' = -π(y1+1)/4

    参数:
        t (float): 自变量(时间/位置)
        y (array): 状态向量 [y1, y2], 其中 y1=u, y2=u'

    返回:
        list: 导数值 [y1', y2']
    """
    dydt = [
        y[1],  # u' = y2
        -np.pi * (y[0] + 1) / 4  # u'' = -π(u+1)/4
    ]
    return dydt


def boundary_conditions_scipy(ya, yb):
    """
    为scipy.solve_bvp定义边界条件

    边界条件: u(0) = 1, u(1) = 1
    ya[0] 应该等于 1, yb[0] 应该等于 1

    参数:
        ya (array): 左边界值 [u(0), u'(0)]
        yb (array): 右边界值 [u(1), u'(1)]

    返回:
        array: 边界条件残差
    """
    return np.array([ya[0] - 1, yb[0] - 1])


def ode_system_scipy(x, y):
    """
    为scipy.solve_bvp定义ODE系统

    参数:
        x (float): 自变量
        y (array): 状态向量 [y1, y2]

    返回:
        array: 导数组成的列向量
    """
    return np.vstack([y[1], -np.pi * (y[0] + 1) / 4])


def solve_bvp_shooting_method(x_span, boundary_conditions, n_points=100, tolerance=1e-6):
    """
    使用打靶法求解边界值问题

    参数:
        x_span (tuple): 求解域 (x_start, x_end)
        boundary_conditions (tuple): (u_left, u_right)
        n_points (int): 离散点的数量
        tolerance (float): 收敛容差

    返回:
        tuple: (x坐标数组, y值数组) 解决方案数组
    """
    x0, x_end = x_span
    ua, ub = boundary_conditions

    x_arr = np.linspace(x0, x_end, n_points)

    def objective_function(m):
        """目标函数：计算在x_end处的解与期望边界的差异"""
        y0 = np.array([ua, float(m)])  # 确保转换为浮点数

        sol = solve_ivp(
            fun=ode_system_shooting,
            t_span=[x0, x_end],
            y0=y0,
            t_eval=x_arr,
            rtol=tolerance
        )

        if not sol.success:
            return float('inf')  # 如果求解失败返回一个大数

        return sol.y[0, -1] - ub

    # 使用fsolve寻找正确的初始斜率
    m_guess = 0.0
    m_solution = fsolve(objective_function, m_guess, xtol=tolerance)[0]

    # 使用找到的斜率求解最终解
    final_y0 = np.array([ua, m_solution])
    sol = solve_ivp(
        fun=ode_system_shooting,
        t_span=[x0, x_end],
        y0=final_y0,
        t_eval=x_arr,
        rtol=tolerance
    )

    if not sol.success:
        raise RuntimeError("打靶法求解失败")

    return sol.t, sol.y[0]


def solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points=50):
    """
    使用scipy.solve_bvp求解边界值问题

    参数:
        x_span (tuple): 求解域 (x_start, x_end)
        boundary_conditions (tuple): (u_left, u_right)
        n_points (int): 初始网格点数

    返回:
        tuple: (x坐标数组, y值数组) 解决方案数组
    """
    x0, x_end = x_span
    ua, ub = boundary_conditions

    x_initial = np.linspace(x0, x_end, n_points)
    y_initial = np.zeros((2, n_points))
    y_initial[0] = 1  # u(x) = 1
    y_initial[1] = 0  # u'(x) = 0

    sol = solve_bvp(
        ode_system_scipy,
        boundary_conditions_scipy,
        x_initial,
        y_initial,
        tol=1e-6
    )

    x_fine = np.linspace(x0, x_end, n_points * 2)
    y_fine = sol.sol(x_fine)

    return x_fine, y_fine[0]


def compare_methods_and_plot(x_span=(0, 1), boundary_conditions=(1, 1), n_points=100):
    """
    比较打靶法和scipy.solve_bvp方法，并生成对比图
    """
    print("使用打靶法求解...")
    x_shoot, u_shoot = solve_bvp_shooting_method(
        x_span, boundary_conditions, n_points=n_points
    )

    print("使用scipy.solve_bvp求解...")
    x_scipy, u_scipy = solve_bvp_scipy_wrapper(
        x_span, boundary_conditions, n_points=n_points // 2
    )

    # 绘制解决方案比较图
    plt.figure(figsize=(10, 6))
    plt.plot(x_shoot, u_shoot, 'b-', linewidth=2, alpha=0.7, label='打靶法')
    plt.plot(x_scipy, u_scipy, 'r--', linewidth=2, alpha=0.7, label='SciPy solve_bvp')
    plt.plot([x_span[0], x_span[1]], [boundary_conditions[0], boundary_conditions[1]],
             'ko', markersize=8)
    plt.title('边界值问题解决方案比较\n$u\'\'(x) = -π(u(x)+1)/4$', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bvp_comparison.png', dpi=150)
    plt.show()

    # 计算并绘制差异
    u_scipy_interp = np.interp(x_shoot, x_scipy, u_scipy)
    diff = np.abs(u_shoot - u_scipy_interp)

    plt.figure(figsize=(10, 6))
    plt.plot(x_shoot, diff, 'g-', linewidth=2, alpha=0.8)
    plt.title('两种解决方案之间的绝对差异', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('|u_shoot - u_scipy|', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bvp_difference.png', dpi=150)
    plt.show()

    print("\n解决方案比较:")
    print(f"最大绝对差异: {np.max(diff):.4e}")
    print(f"平均绝对差异: {np.mean(diff):.4e}")
    print(f"均方根误差: {np.sqrt(np.mean(diff ** 2)):.4e}")

    return {
        'x_shoot': x_shoot,
        'u_shoot': u_shoot,
        'x_scipy': x_scipy,
        'u_scipy': u_scipy,
        'max_diff': np.max(diff),
        'mean_diff': np.mean(diff),
        'rmse': np.sqrt(np.mean(diff ** 2))
    }


if __name__ == "__main__":
    # 运行比较和绘图
    results = compare_methods_and_plot(n_points=100)

    # 保存结果
    np.savez('bvp_solutions.npz',
             x_shoot=results['x_shoot'],
             u_shoot=results['u_shoot'],
             x_scipy=results['x_scipy'],
             u_scipy=results['u_scipy'])

    print("\n项目成功完成!")
    print(f"平均误差: {results['mean_diff']:.2e}")
    print(f"最大误差: {results['max_diff']:.2e}")
