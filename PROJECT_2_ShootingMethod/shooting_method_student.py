#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目：打靶法求解边值问题
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_bvp
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

def ode_system_shooting(y, t):
    """
    为打靶法定义ODE系统：u'' = -π(u+1)/4 -> 转化为一阶系统 y1 = u, y2 = u'
    参数:
        y (array): 状态向量 [y1, y2]
        t (float): 自变量(时间/位置)
    返回:
        list: 导数向量 [y1', y2']
    """
    try:
        return [y[1], -np.pi * (y[0] + 1) / 4]
    except Exception as e:
        raise RuntimeError(f"计算ODE系统时出错: {str(e)}")

def boundary_conditions_scipy(ya, yb):
    """
    为scipy.solve_bvp定义边界条件：u(0) = 1, u(1) = 1
    参数:
        ya (array): 左边界值 [u(0), u'(0)]
        yb (array): 右边界值 [u(1), u'(1)]
    返回:
        array: 边界条件残差
    """
    try:
        return np.array([ya[0] - 1, yb[0] - 1])
    except Exception as e:
        raise RuntimeError(f"计算边界条件时出错: {str(e)}")

def ode_system_scipy(x, y):
    """
    scipy.solve_bvp需要的ODE系统格式
    参数:
        x: 自变量
        y: 状态向量
    返回:
        array: 导数组成的列向量
    """
    try:
        return np.vstack([y[1], -np.pi * (y[0] + 1) / 4])
    except Exception as e:
        raise RuntimeError(f"计算scipy ODE系统时出错: {str(e)}")

def solve_bvp_shooting_method(x_span, boundary_conditions, n_points=100, max_iterations=10, tolerance=1e-6):
    """
    使用打靶法求解边值问题
    参数:
        x_span (tuple): 求解区间 (x_start, x_end)
        boundary_conditions (tuple): 边界条件 (u_left, u_right)
        n_points (int): 离散点数量
        max_iterations (int): 最大迭代次数
        tolerance (float): 容差
    返回:
        tuple: (x坐标数组, y值数组)
    """
    # 输入验证
    if not isinstance(x_span, (tuple, list)) or len(x_span) != 2:
        raise ValueError("x_span必须是长度为2的元组或列表")
    
    x_start, x_end = x_span
    if x_start >= x_end:
        raise ValueError("x_start必须小于x_end")
    
    if not isinstance(boundary_conditions, (tuple, list)) or len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions必须是长度为2的元组或列表")
    
    if not isinstance(n_points, int) or n_points < 10:
        raise ValueError("n_points必须是整数且至少为10")
    
    if not isinstance(tolerance, (int, float)) or tolerance <= 0:
        raise ValueError("tolerance必须是正数")
    
    u_left, u_right = boundary_conditions
    x = np.linspace(x_start, x_end, n_points)
    
    # 初始猜测
    m1 = -1.0  # 第一个猜测
    y0 = [u_left, m1]
    
    # 解第一个初始值问题
    sol1 = odeint(ode_system_shooting, y0, x)
    u_end_1 = sol1[-1, 0]  # u(x_end)的值
    
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
    
    # 迭代改善
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
    
    # 未收敛，返回最佳结果
    print(f"警告：打靶法在 {max_iterations} 次迭代后未收敛。最终边界误差: {abs(u_end_3 - u_right):.2e}")
    return x, sol3[:, 0]

def solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points=50):
    """
    使用scipy.solve_bvp求解边值问题
    参数:
        x_span (tuple): 求解区间 (x_start, x_end)
        boundary_conditions (tuple): 边界条件 (u_left, u_right)
        n_points (int): 初始化网格点数
    返回:
        tuple: (x坐标数组, y值数组)
    """
    # 输入验证
    if not isinstance(x_span, (tuple, list)) or len(x_span) != 2:
        raise ValueError("x_span必须是长度为2的元组或列表")
    
    x_start, x_end = x_span
    if x_start >= x_end:
        raise ValueError("x_start必须小于x_end")
    
    if not isinstance(boundary_conditions, (tuple, list)) or len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions必须是长度为2的元组或列表")
    
    if not isinstance(n_points, int) or n_points < 5:
        raise ValueError("n_points必须是整数且至少为5")
    
    u_left, u_right = boundary_conditions
    
    # 创建初始网格
    x_init = np.linspace(x_start, x_end, n_points)
    
    # 初始猜测：边界值之间的线性插值
    y_init = np.zeros((2, x_init.size))
    y_init[0] = u_left + (u_right - u_left) * (x_init - x_start) / (x_end - x_start)
    y_init[1] = (u_right - u_left) / (x_end - x_start)  # 恒定斜率猜测
    
    try:
        # 使用scipy.solve_bvp求解
        sol = solve_bvp(ode_system_scipy, boundary_conditions_scipy, x_init, y_init, tol=1e-6)
        
        if not sol.success:
            raise RuntimeError(f"scipy.solve_bvp 求解失败: {sol.message}")
        
        # 在细网格上生成解决方案
        x_fine = np.linspace(x_start, x_end, 100)
        y_fine = sol.sol(x_fine)[0]
        
        return x_fine, y_fine
        
    except Exception as e:
        raise RuntimeError(f"scipy.solve_bvp 中发生错误: {str(e)}")

def compare_methods_and_plot(x_span=(0, 1), boundary_conditions=(1, 1), n_points=100):
    """
    比较打靶法和scipy.solve_bvp的结果，并绘制图像
    返回：
        dict: 包含解决方案和分析结果的字典
    """
    # 输入验证
    if not isinstance(x_span, (tuple, list)) or len(x_span) != 2:
        raise ValueError("x_span必须是长度为2的元组或列表")
    
    x_start, x_end = x_span
    if x_start >= x_end:
        raise ValueError("x_start必须小于x_end")
    
    if not isinstance(boundary_conditions, (tuple, list)) or len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions必须是长度为2的元组或列表")
    
    if not isinstance(n_points, int) or n_points < 10:
        raise ValueError("n_points必须是整数且至少为10")
    
    u_left, u_right = boundary_conditions
    
    print("使用打靶法求解...")
    x_shoot, u_shoot = solve_bvp_shooting_method(x_span, boundary_conditions, n_points)
    
    print("使用scipy.solve_bvp求解...")
    x_scipy, u_scipy = solve_bvp_scipy_wrapper(x_span, boundary_conditions, n_points // 2)
    
    # 绘制解决方案比较图
    plt.figure(figsize=(10, 6))
    plt.plot(x_shoot, u_shoot, 'b-', lw=2, alpha=0.7, label='打靶法')
    plt.plot(x_scipy, u_scipy, 'r--', lw=2, alpha=0.7, label='SciPy solve_bvp')
    plt.plot([x_span[0], x_span[1]], [boundary_conditions[0], boundary_conditions[1]], 
             'ko', markersize=8)
    plt.title('边值问题解决方案比较\n$u\'\'(x) = -π(u(x)+1)/4$', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bvp_comparison.png', dpi=150)
    plt.show()
    
    # 计算差异
    u_scipy_interp = np.interp(x_shoot, x_scipy, u_scipy)
    diff = np.abs(u_shoot - u_scipy_interp)
    
    # 绘制差异图
    plt.figure(figsize=(10, 6))
    plt.plot(x_shoot, diff, 'g-', lw=2, alpha=0.8)
    plt.title('两种方法的绝对差异', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('|u_打靶 - u_scipy|', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bvp_difference.png', dpi=150)
    plt.show()
    
    # 计算最大差异和均方根误差
    max_diff = np.max(diff)
    rms_diff = np.sqrt(np.mean(diff ** 2))
    
    # 准备返回结果
    results = {
        'x_shooting': x_shoot,
        'y_shooting': u_shoot,
        'x_scipy': x_scipy,
        'y_scipy': u_scipy,
        'max_difference': max_diff,
        'rms_difference': rms_diff,
        'boundary_error_shooting': [abs(u_shoot[0] - u_left), abs(u_shoot[-1] - u_right)],
        'boundary_error_scipy': [abs(u_scipy[0] - u_left), abs(u_scipy[-1] - u_right)]
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
