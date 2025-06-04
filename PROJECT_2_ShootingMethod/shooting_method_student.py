#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目2：打靶法与scipy.solve_bvp求解边值问题

本项目要求实现打靶法和scipy.solve_bvp两种方法来求解二阶线性常微分方程边值问题：
u''(x) = -π(u(x)+1)/4
边界条件：u(0) = 1, u(1) = 1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import fsolve
import warnings
import matplotlib
import sys

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
        y[1],                 # u' = y2
        -np.pi*(y[0]+1)/4    # u'' = -π(u+1)/4
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
    return np.vstack([y[1], -np.pi*(y[0]+1)/4])


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
    # 输入验证
    if not isinstance(x_span, (tuple, list)) or len(x_span) != 2:
        raise ValueError("x_span should be a tuple of (start, end)")
    if not isinstance(boundary_conditions, (tuple, list)) or len(boundary_conditions) != 2:
        raise ValueError("boundary_conditions should be a tuple of (left, right)")
    
    x0, x_end = x_span
    ua, ub = boundary_conditions
    
    if x0 >= x_end:
        raise ValueError("x_start must be less than x_end")
    
    x_arr = np.linspace(x0, x_end, n_points)
    
    def objective_function(m):
        """目标函数：计算在x_end处的解与期望边界的差异"""
        y0 = np.array([ua, m])  # 移除了float转换以避免警告
        
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
    
    x_fine = np.linspace(x0, x_end, n_points*2)
    y_fine = sol.sol(x_fine)
    
    return x_fine, y_fine[0]


def compare_methods_and_plot(x_span=(0, 1), boundary_conditions=(1, 1), n_points=100):
    """
    比较打靶法和scipy.solve_bvp方法，并生成对比图
    
    返回:
        dict: 包含解决方案和差异的字典，键名符合测试要求
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
        x_span, boundary_conditions, n_points=n_points//2
    )
    
    # 计算差异
    u_scipy_interp = np.interp(x_shoot, x_scipy, u_scipy)
    diff = np.abs(u_shoot - u_scipy_interp)
    
    # 确保返回的字典键名与测试要求一致
    results = {
        'x_shooting': x_shoot,
        'y_shooting': u_shoot,
        'x_scipy': x_scipy,
        'y_scipy': u_scipy,
        'max_difference': np.max(diff),
        'rms_difference': np.sqrt(np.mean(diff**2))
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
