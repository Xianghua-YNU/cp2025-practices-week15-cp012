import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.linalg import solve

# ============================================================================

# 方法1：有限差分法 (Finite Difference Method)
# ============================================================================

def solve_bvp_finite_difference(n):
    """
    使用有限差分法求解二阶常微分方程边值问题。
    """
    # Step 1: 创建网格
    h = 5.0 / (n + 1)  # 步长
    x_grid = np.linspace(0, 5, n + 2)  # 创建包含边界的网格点
    
    # Step 2: 构建系数矩阵 A 和右端向量 b
    A = np.zeros((n, n))  # 初始化系数矩阵
    b = np.zeros(n)       # 初始化右端向量
    
    # Step 3: 填充矩阵 A 和向量 b
    for i in range(n):
        x_i = x_grid[i + 1]  # 当前内部点的 x 坐标
        
        # 计算差分系数
        coeff_left = 1.0 / h**2 - np.sin(x_i) / (2.0 * h)
        coeff_center = -2.0 / h**2 + np.exp(x_i)
        coeff_right = 1.0 / h**2 + np.sin(x_i) / (2.0 * h)
        
        # 填充系数矩阵 A
        if i > 0:
            A[i, i-1] = coeff_left
        A[i, i] = coeff_center
        if i < n - 1:
            A[i, i+1] = coeff_right
        
        # 填充右端向量 b
        b[i] = x_i**2
        
        # 处理边界条件
        if i == 0:  # 左边界
            b[i] -= coeff_left * 0.0  # y(0) = 0
        if i == n - 1:  # 右边界
            b[i] -= coeff_right * 3.0  # y(5) = 3
    
    # Step 4: 求解线性系统
    y_interior = solve(A, b)  # 解内部点的 y 值
    
    # Step 5: 组合完整解
    y_solution = np.zeros(n + 2)
    y_solution[0] = 0.0       # 左边界值
    y_solution[1:-1] = y_interior  # 内部点解
    y_solution[-1] = 3.0      # 右边界值
    
    return x_grid, y_solution

# ============================================================================

# 方法2：scipy.integrate.solve_bvp 方法
# ============================================================================

def ode_system_for_solve_bvp(x, y):
    """
    定义ODE系统，将二阶ODE转换为一阶系统。
    """
    y0 = y[0]  # y(x)
    y1 = y[1]  # y'(x)
    
    dy0_dx = y1  # dy/dx = y'
    dy1_dx = -np.sin(x) * y1 - np.exp(x) * y0 + x**2  # d(y')/dx
    
    return np.vstack([dy0_dx, dy1_dx])

def boundary_conditions_for_solve_bvp(ya, yb):
    """
    定义边界条件。
    """
    return np.array([ya[0] - 0, yb[0] - 3])  # y(0)=0 和 y(5)=3

def solve_bvp_scipy(n_initial_points=11):
    """
    使用 scipy.integrate.solve_bvp 求解BVP。
    """
    # Step 1: 创建初始网格
    x_initial = np.linspace(0, 5, n_initial_points)
    
    # Step 2: 创建初始猜测
    y_initial = np.zeros((2, n_initial_points))
    y_initial[0] = np.linspace(0, 3, n_initial_points)  # y(x) 的初始猜测
    y_initial[1] = np.ones(n_initial_points) * 0.6      # y'(x) 的初始猜测
    
    # Step 3: 调用 solve_bvp
    solution = solve_bvp(ode_system_for_solve_bvp, boundary_conditions_for_solve_bvp, 
                         x_initial, y_initial)
    
    # Step 4: 提取解
    if solution.success:
        return solution.x, solution.y[0]  # 返回 x 和 y(x)
    else:
        raise RuntimeError("solve_bvp failed to converge")

# ============================================================================

# 主程序：演示两种方法求解边值问题
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("二阶常微分方程边值问题求解演示")
    print("方程: y''(x) + sin(x)*y'(x) + exp(x)*y(x) = x^2")
    print("边界条件: y(0) = 0, y(5) = 3")
    print("=" * 80)
    
    # 定义问题参数
    x_start, y_start = 0.0, 0.0  # 左边界条件
    x_end, y_end = 5.0, 3.0      # 右边界条件
    num_points = 100             # 离散点数
    
    print(f"\n求解区间: [{x_start}, {x_end}]")
    print(f"边界条件: y({x_start}) = {y_start}, y({x_end}) = {y_end}")
    print(f"离散点数: {num_points}")
    
    # ========================================================================

    # 方法1：有限差分法
    # ========================================================================
    print("\n" + "-" * 60)
    print("方法1：有限差分法 (Finite Difference Method)")
    print("-" * 60)
    
    try:
        x_fd, y_fd = solve_bvp_finite_difference(num_points - 2)  # 减去边界点
        print("有限差分法求解成功！")
    except Exception as e:
        print(f"有限差分法求解失败: {e}")
        x_fd, y_fd = None, None
    
    # ========================================================================

    # 方法2：scipy.integrate.solve_bvp
    # ========================================================================
    print("\n" + "-" * 60)
    print("方法2：scipy.integrate.solve_bvp")
    print("-" * 60)
    
    try:
        x_scipy, y_scipy = solve_bvp_scipy(num_points)
        print("solve_bvp 求解成功！")
    except Exception as e:
        print(f"solve_bvp 求解失败: {e}")
        x_scipy, y_scipy = None, None
    
    # ========================================================================

    # 结果可视化与比较
    # ========================================================================
    print("\n" + "-" * 60)
    print("结果可视化与比较")
    print("-" * 60)
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制两种方法的解
    if x_fd is not None and y_fd is not None:
        plt.plot(x_fd, y_fd, 'b-', linewidth=2, label='Finite Difference Method', alpha=0.8)
    
    if x_scipy is not None and y_scipy is not None:
        plt.plot(x_scipy, y_scipy, 'r--', linewidth=2, label='scipy solve_bvp', alpha=0.8)
    
    # 标记边界条件
    plt.scatter([x_start, x_end], [y_start, y_end], 
               color='red', s=100, zorder=5, label='Boundary Conditions')
    
    # 图形美化
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y(x)', fontsize=12)
    plt.title(r"BVP Solution: $y'' + \sin(x)y' + e^x y = x^2$, $y(0)=0$, $y(5)=3$", 
              fontsize=14, pad=20)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 显示图形
    plt.show()
    
    # ========================================================================

    # 数值结果比较
    # ========================================================================
    print("\n" + "-" * 60)
    print("数值结果比较")
    print("-" * 60)
    
    # 在几个特定点比较解的值
    test_points = [1.0, 2.5, 4.0]
    
    for x_test in test_points:
        print(f"\n在 x = {x_test} 处的解值:")
        
        if x_fd is not None and y_fd is not None:
            y_test_fd = np.interp(x_test, x_fd, y_fd)
            print(f"  有限差分法:  {y_test_fd:.6f}")
        
        if x_scipy is not None and y_scipy is not None:
            y_test_scipy = np.interp(x_test, x_scipy, y_scipy)
            print(f"  solve_bvp:   {y_test_scipy:.6f}")
    
    print("\n" + "=" * 80)
    print("求解完成！")
    print("=" * 80)
