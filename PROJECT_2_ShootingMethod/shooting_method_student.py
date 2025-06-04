#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：打靶法与scipy.solve_bvp求解边值问题
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_bvp
from scipy.optimize import fsolve
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

def ode_system_shooting(t, y):
    """
    为打靶法定义ODE系统：u'' = -π(u+1)/4 -> 转化为一阶系统 y1 = u, y2 = u'
    """
    return [y[1], -np.pi*(y[0] + 1)/4]

def boundary_conditions_scipy(ya, yb):
    """
    为scipy.solve_bvp定义边界条件：u(0) = 1, u(1) = 1
    """
    return np.array([ya[0] - 1, yb[0] - 1])

def ode_system_scipy(x, y):
    """
    scipy.solve_bvp需要的ODE系统格式
    """
    return np.vstack([y[1], -np.pi*(y[0] + 1)/4])

def solve_bvp_shooting_method(x_span, boundary_conditions, n_points=100, max_iterations=10, tolerance=1e-6):
    """
    使用打靶法求解边值问题
    """
    # 输入验证
    if len(x_span) != 2 or x_span[1] <= x_span[0]:
        raise ValueError("x_span必须是(x_start, x_end)，且x_end > x_start")
    if len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions必须是(u_left, u_right)的元组")
    if n_points < 10:
        raise ValueError("n_points至少要10个点")
    
    x_start, x_end = x_span
    u_left, u_right = boundary_conditions
    
    x = np.linspace(x_start, x_end, n_points)
    
    # 初始猜测
    m1 = -1.0  # 第一个猜测
    y0 = [u_left, m1]
    
    # 解第一个初始值问题
    sol1 = odeint(ode_system_shooting, y0, x)
    u_end_1 = sol1[-1, 0]
    
    # 检查是否满足精度
    if abs(u_end_1 - u_right) < tolerance:
        return x, sol1[:, 0]
    
    # 第二个猜测
    if abs(u_end_1) > 1e-12:
        m2 = m1 * u_right / u_end_1
    else:
        m2 = m1 + 1.0
    y0 = [u_left, m2]
    sol2 = odeint(ode_system_shooting, y0, x)
    u_end_2 = sol2[-1, 0]
    
    if abs(u_end_2 - u_right) < tolerance:
        return x, sol2[:, 0]
    
    for iteration in range(max_iterations):
        # 避免除以零
        if abs(u_end_2 - u_end_1) < 1e-12:
            m_increment = 0.1
            m3 = m2 + m_increment
        else:
            m3 = m2 + (u_right - u_end_2) * (m2 - m1) / (u_end_2 - u_end_1)
        
        y0 = [u_left, m3]
        sol3 = odeint(ode_system_shooting, y0, x)
        u_end_3 = sol3[-1, 0]
        
        if abs(u_end_3 - u_right) < tolerance:
            return x, sol3[:, 0]
        
        m1, m2 = m2, m3
        u_end_1, u_end_2 = u_end_2, u_end_3
    
    # 未收敛，返回最好结果
    print(f"警告：打靶法在 {max_iterations} 次迭代后未收敛。")
    print(f"最终边界误差：{abs(u_end_3 - u_right):.2e}")
    return x, sol3[:, 0]

def solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points=50):
    """
    使用scipy.solve_bvp求解边值问题
    """
    # 输入验证
    if len(x_span) != 2 or x_span[1] <= x_span[0]:
        raise ValueError("x_span必须是(x_start, x_end)，且x_end > x_start")
    if len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions必须是(u_left, u_right)的元组")
    if n_points < 5:
        raise ValueError("n_points至少要5个点")
    
    x_start, x_end = x_span
    u_left, u_right = boundary_conditions
    
    x_init = np.linspace(x_start, x_end, n_points)
    y_init = np.zeros((2, n_points))
    y_init[0] = u_left  # 初始猜测：u(x) = u_left
    y_init[1] = 0  # 初始猜测：u’(x) = 0
    
    sol = solve_bvp(ode_system_scipy, boundary_conditions_scipy, x_init, y_init, tol=1e-6)
    
    x_fine = np.linspace(x_start, x_end, n_points * 2)
    y_fine = sol.sol(x_fine)[0]
    
    return x_fine, y_fine

def compare_methods_and_plot(x_span=(0, 1), boundary_conditions=(1, 1), n_points=100):
    """
    比较打靶法和scipy.solve_bvp的结果，并绘制图像
    """
    # 输入验证
    if len(x_span) != 2 or x_span[1] <= x_span[0]:
        raise ValueError("x_span必须是(x_start, x_end)，且x_end > x_start")
    if len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions必须是(u_left, u_right)的元组")
    if n_points < 10:
        raise ValueError("n_points至少要10个点")
    
    print("使用打靶法求解...")
    x_shoot, u_shoot = solve_bvp_shooting_method(x_span, boundary_conditions, n_points)
    
    print("使用scipy.solve_bvp求解...")
    x_scipy, u_scipy = solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points // 2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_shoot, u_shoot, 'b-', lw=2, alpha=0.7, label='打靶法')
    plt.plot(x_scipy, u_scipy, 'r--', lw=2, alpha=0.7, label='SciPy solve_bvp')
    plt.plot([x_span[0], x_span[1]], [boundary_conditions[0], boundary_conditions[1]], 
             'ko', markersize=8)
    plt.title('边值问题解决方案比较\n$u''(x) = -π(u(x)+1)/4$', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bvp_comparison.png', dpi=150)
    plt.show()
    
    u_scipy_interp = np.interp(x_shoot, x_scipy, u_scipy)
    diff = np.abs(u_shoot - u_scipy_interp)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_shoot, diff, 'g-', lw=2, alpha=0.8)
    plt.title('两种方法的绝对差异', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('|u_打靶 - u_scipy|', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bvp_difference.png', dpi=150)
    plt.show()
    
    results = {
        'x_shooting': x_shoot,
        'y_shooting': u_shoot,
        'x_scipy': x_scipy,
        'y_scipy': u_scipy,
        'max_difference': np.max(diff),
        'rms_difference': np.sqrt(np.mean(diff ** 2)),
        'boundary_error_shooting': [abs(u_shoot[0] - boundary_conditions[0]), 
                                   abs(u_shoot[-1] - boundary_conditions[1])],
        'boundary_error_scipy': [abs(u_scipy[0] - boundary_conditions[0]), 
                                abs(u_scipy[-1] - boundary_conditions[1])]
    }
    
    return results

if __name__ == "__main__":
    results = compare_methods_and_plot(n_points=100)
    np.savez('bvp_solutions.npz', 
             x_shooting=results['x_shooting'], 
             y_shooting=results['y_shooting'],
             x_scipy=results['x_scipy'],
             y_scipy=results['y_scipy'])
    
    print("\n项目成功完成!")
    print(f"最大差异: {results['max_difference']:.2e}")
    print(f"均方根误差: {results['rms_difference']:.2e}")
