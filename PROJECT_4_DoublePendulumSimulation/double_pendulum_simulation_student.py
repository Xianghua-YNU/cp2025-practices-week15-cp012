import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.animation as animation

# 可以在函数中使用的常量
G_CONST = 9.81  # 重力加速度 (m/s^2)
L_CONST = 0.4  # 每个摆臂的长度 (m)
M_CONST = 1.0  # 每个摆锤的质量 (kg)


def derivatives(y, t, L1, L2, m1, m2, g):
    """
    返回双摆状态向量y的时间导数。
    """
    theta1, omega1, theta2, omega2 = y

    dtheta1_dt = omega1
    dtheta2_dt = omega2

    # 计算公共分母
    common_denominator = 3 - np.cos(2 * theta1 - 2 * theta2)

    # 计算 domega1_dt 的分子
    domega1_numerator = - (
            omega1 ** 2 * np.sin(2 * theta1 - 2 * theta2) +
            2 * omega2 ** 2 * np.sin(theta1 - theta2) +
            (g / L1) * (np.sin(theta1 - 2 * theta2) + 3 * np.sin(theta1))
    )

    # 计算 domega2_dt 的分子
    domega2_numerator = (
            4 * omega1 ** 2 * np.sin(theta1 - theta2) +
            omega2 ** 2 * np.sin(2 * theta1 - 2 * theta2) +
            2 * (g / L1) * (np.sin(2 * theta1 - theta2) - np.sin(theta2))
    )

    domega1_dt = domega1_numerator / common_denominator
    domega2_dt = domega2_numerator / common_denominator

    return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]


def solve_double_pendulum(initial_conditions, t_span, t_points, L_param=L_CONST, g_param=G_CONST):
    """
    使用 odeint 求解双摆的常微分方程组。
    """
    # 构建初始状态向量
    y0 = [
        initial_conditions['theta1'],
        initial_conditions['omega1'],
        initial_conditions['theta2'],
        initial_conditions['omega2']
    ]

    # 生成时间数组
    t_arr = np.linspace(t_span[0], t_span[1], t_points)

    # 调用 odeint 求解，设置严格的容差以保证能量守恒
    sol_arr = odeint(
        derivatives, y0, t_arr,
        args=(L_param, L_param, M_CONST, M_CONST, g_param),
        rtol=1e-8, atol=1e-8
    )

    return t_arr, sol_arr


def calculate_energy(sol_arr, L_param=L_CONST, m_param=M_CONST, g_param=G_CONST):
    """
    计算双摆系统的总能量 (动能 + 势能)。
    """
    theta1 = sol_arr[:, 0]
    omega1 = sol_arr[:, 1]
    theta2 = sol_arr[:, 2]
    omega2 = sol_arr[:, 3]

    # 计算势能
    potential_energy = -m_param * g_param * L_param * (2 * np.cos(theta1) + np.cos(theta2))

    # 计算动能
    kinetic_energy = m_param * L_param ** 2 * (
            omega1 ** 2 + 0.5 * omega2 ** 2 + omega1 * omega2 * np.cos(theta1 - theta2)
    )

    return kinetic_energy + potential_energy


# --- 可选任务: 动画 ---
def animate_double_pendulum(t_arr, sol_arr, L_param=L_CONST, skip_frames=10):
    """
    创建双摆的动画。
    """
    theta1_all = sol_arr[:, 0]
    theta2_all = sol_arr[:, 2]

    # 选择动画帧
    frame_indices = np.arange(0, len(t_arr), skip_frames)
    theta1_anim = theta1_all[frame_indices]
    theta2_anim = theta2_all[frame_indices]
    t_anim = t_arr[frame_indices]

    # 转换为笛卡尔坐标
    x1 = L_param * np.sin(theta1_anim)
    y1 = -L_param * np.cos(theta1_anim)
    x2 = x1 + L_param * np.sin(theta2_anim)
    y2 = y1 - L_param * np.cos(theta2_anim)

    # 创建图形和坐标轴
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, autoscale_on=False,
                         xlim=(-2 * L_param, 2 * L_param),
                         ylim=(-2 * L_param, 0.1))
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Double Pendulum Animation')

    # 初始化绘图元素
    line, = ax.plot([], [], 'o-', lw=2, markersize=10, color='red')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]
        line.set_data(thisx, thisy)
        time_text.set_text(f'Time = {t_anim[i]:.1f}s')
        return line, time_text

    ani = animation.FuncAnimation(
        fig, animate, frames=len(frame_indices),
        interval=50, blit=True, init_func=init
    )

    return ani


if __name__ == '__main__':
    # 初始条件（90度垂直向下）
    initial_conditions = {
        'theta1': np.pi / 2,  # 90度（垂直向右）
        'omega1': 0.0,
        'theta2': np.pi / 2,  # 90度（垂直向右）
        'omega2': 0.0
    }

    t_span = (0, 20)  # 模拟时长20秒
    t_points = 2000  # 时间点数

    # 求解微分方程
    t_arr, sol_arr = solve_double_pendulum(initial_conditions, t_span, t_points)

    # 计算能量
    energy = calculate_energy(sol_arr)

    # 绘制能量图
    plt.figure(figsize=(10, 5))
    plt.plot(t_arr, energy, label='Total Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.title('Energy Conservation of Double Pendulum')
    plt.grid(True)
    plt.legend()
    plt.show()

    # 显示能量统计
    initial_energy = energy[0]
    energy_variation = np.max(energy) - np.min(energy)
    print(f"Initial Energy: {initial_energy:.4f} J")
    print(f"Energy Variation: {energy_variation:.2e} J")
anim = animate_double_pendulum(t_arr, sol_arr, skip_frames=10)
plt.show()
