#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：打靶法与scipy.solve_bvp求解边值问题
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import fsolve
import warnings
import matplotlib
import sys

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


def ode_system_shooting(t, y):
    """
    为打靶法定义ODE系统
    """
    return [y[1], -np.pi * (y[0] + 1) / 4]


def boundary_conditions_scipy(ya, yb):
    """
    为scipy.solve_bvp定义边界条件
    """
    return np.array([ya[0] - 1, yb[0] - 1])


def ode_system_scipy(x, y):
    """
    为scipy.solve_bvp定义ODE系统
    """
    return np.vstack([y[1], -np.pi * (y[0] + 1) / 4])


def solve_bvp_shooting_method(x_span, boundary_conditions, n_points=100, tolerance=1e-6):
    """
    使用打靶法求解边界值问题
    """
    # 输入验证
    if not isinstance(x_span, (tuple, list)) or len(x_span) != 2:
        raise ValueError("x_span should be a tuple of (start, end)")
    if not isinstance(boundary_conditions, (tuple, list)) or len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions should be a tuple of (left, right)")

    x0, x_end = x_span
    ua, ub = boundary_conditions

    if x0 >= x_end:
        raise ValueError("x_start must be less than x_end")

    x_eval = np.linspace(x0, x_end, n_points)

    def objective_function(m):
        """目标函数：计算在x_end处的解与期望边界的差异"""
        # 确保m是标量值
        m = float(m[0] if isinstance(m, (np.ndarray, list)) else m)
        y0 = np.array([ua, m])

        sol = solve_ivp(
            fun=ode_system_shooting,
            t_span=[x0, x_end],
            y0=y0,
            t_eval=x_eval,
            rtol=tolerance
        )

        if not sol.success:
            return float('inf')

        return sol.y[0, -1] - ub

    # 使用fsolve寻找正确的初始斜率
    m_guess = np.array([0.0])  # 使用数组作为初始猜测
    m_solution = fsolve(objective_function, m_guess, xtol=tolerance)

    # 使用找到的斜率求解最终解
    final_y0 = np.array([ua, float(m_solution[0])])
    sol = solve_ivp(
        fun=ode_system_shooting,
        t_span=[x0, x_end],
        y0=final_y0,
        t_eval=x_eval,
        rtol=tolerance
    )

    if not sol.success:
        raise RuntimeError("打靶法求解失败")

    return sol.t, sol.y[0]


def solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points=50):
    """
    使用scipy.solve_bvp求解边界值问题
    """
    # 输入验证
    if not isinstance(x_span, (tuple, list)) or len(x_span) != 2:
        raise ValueError("x_span should be a tuple of (start, end)")
    if not isinstance(boundary_conditions, (tuple, list)) or len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions should be a tuple of (left, right)")

    x0, x_end = x_span
    ua, ub = boundary_conditions

    if x0 >= x_end:
        raise ValueError("x_start must be less than x_end")

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
    # 输入验证
    if not isinstance(x_span, (tuple, list)) or len(x_span) != 2:
        raise ValueError("x_span should be a tuple of (start, end)")
    if not isinstance(boundary_conditions, (tuple, list)) or len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions should be a tuple of (left, right)")

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

    # 确保返回的字典键名与测试要求一致
    results = {
        'x_shooting': x_shoot,
        'y_shooting': u_shoot,
        'x_scipy': x_scipy,
        'y_scipy': u_scipy,
        'max_difference': np.max(diff),
        'rms_difference': np.sqrt(np.mean(diff ** 2))
    }

    return results


if __name__ == "__main__":
    # 运行比较和绘图
    results = compare_methods_and_plot(n_points=100)

    # 保存结果
    np.savez('bvp_solutions.npz',
             x_shooting=results['x_shooting'],
             y_shooting=results['y_shooting'],
             x_scipy=results['x_scipy'],
             y_scipy=results['y_scipy'])

    print("\n项目成功完成!")
    print(f"最大差异: {results['max_difference']:.2e}")
    print(f"均方根误差: {results['rms_difference']:.2e}")
