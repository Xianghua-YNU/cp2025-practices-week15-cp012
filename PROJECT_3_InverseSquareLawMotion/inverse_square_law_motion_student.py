"""
学生模板：平方反比引力场中的运动
文件：inverse_square_law_motion_student.py
作者：[你的名字]
日期：[完成日期]

重要：函数名称、参数名称和返回值的结构必须与参考答案保持一致！
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

GM=1
def derivatives(t, state_vector, gm_val):
    """
    计算状态向量 [x, y, vx, vy] 的导数。
    """
    x, y, vx, vy = state_vector
    r_cubed = (x ** 2 + y ** 2) ** 1.5
    if r_cubed < 1e-12:
        ax = -gm_val * x / (1e-12) if x != 0 else 0
        ay = -gm_val * y / (1e-12) if y != 0 else 0
        return [vx, vy, ax, ay]


    ax = -gm_val * x / r_cubed
    ay = -gm_val * y / r_cubed

    return [vx, vy, ax, ay]


def solve_orbit(initial_conditions, t_span, t_eval, gm_val):
    """
    使用 scipy.integrate.solve_ivp 求解轨道运动问题。
    """
    sol = solve_ivp(
        fun=derivatives,
        t_span=t_span,
        y0=initial_conditions,
        t_eval=t_eval,
        args=(gm_val,),
        method='RK45',
        rtol=1e-7,
        atol=1e-9
    )
    return sol


def calculate_energy(state_vector, gm_val, m=1.0):
    """
    计算质点的（比）机械能。
    """
    if state_vector.ndim == 1:
        x, y, vx, vy = state_vector
    else:
        x, y, vx, vy = state_vector.T

    r = np.sqrt(x ** 2 + y ** 2)
    epsilon = 1e-10
    r = np.where(r < epsilon, epsilon, r)

    v_squared = vx ** 2 + vy ** 2
    kinetic_energy_per_m = 0.5 * v_squared
    potential_energy_per_m = -gm_val / r
    specific_energy = kinetic_energy_per_m + potential_energy_per_m

    return specific_energy * m


def calculate_angular_momentum(state_vector, m=1.0):
    """
    计算质点的（比）角动量 (z分量)。
    """
    # Handle both 1D (single state) and 2D (multiple states) cases
    if state_vector.ndim == 1:
        x, y, vx, vy = state_vector
    else:
        x, y, vx, vy = state_vector.T

    specific_Lz = x * vy - y * vx
    return specific_Lz * m


if __name__ == "__main__":
    print("平方反比引力场中的运动 - 学生模板")

    # 示例：设置椭圆轨道初始条件
    ic_ellipse_demo = [1.0, 0.0, 0.0, 0.8]  # 初始位置(1,0)，初始速度(0,0.8)
    t_start_demo = 0
    t_end_demo = 20
    t_eval_demo = np.linspace(t_start_demo, t_end_demo, 500)

    try:
        sol_ellipse = solve_orbit(ic_ellipse_demo, (t_start_demo, t_end_demo), t_eval_demo, gm_val=GM)
        x_ellipse, y_ellipse = sol_ellipse.y[0], sol_ellipse.y[1]

        # 计算能量和角动量
        energy = calculate_energy(sol_ellipse.y.T, GM)
        angular_momentum = calculate_angular_momentum(sol_ellipse.y.T)
        print(f"Ellipse Demo: Initial Energy approx {energy[0]:.4f}, Angular Momentum {angular_momentum[0]:.4f}")

        plt.figure(figsize=(8, 6))
        plt.plot(x_ellipse, y_ellipse, label='椭圆轨道 (示例)')
        plt.plot(0, 0, 'ko', markersize=8, label='中心天体')
        plt.title('轨道运动示例')
        plt.xlabel('x 坐标')
        plt.ylabel('y 坐标')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    except Exception as e:
        print(f"运行示例时发生错误: {e}")

    # 学生需要根据“项目说明.md”完成以下任务：
    # 1. 实现 `derivatives`, `solve_orbit`, `calculate_energy`, `calculate_angular_momentum` 函数。
    # 2. 针对 E > 0, E = 0, E < 0 三种情况设置初始条件，求解并绘制轨道。
    # 3. 针对 E < 0 且固定时，改变角动量，求解并绘制轨道。
    # 4. (可选) 进行坐标转换和对称性分析。

    print("\n请参照 '项目说明.md' 完成各项任务。")
    print("使用 'tests/test_inverse_square_law_motion.py' 文件来测试你的代码实现。")

    pass # 学生代码的主要部分应在函数内实现
