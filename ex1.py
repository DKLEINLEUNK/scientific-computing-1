import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def initial_condition(x, case):
    if case == 1:
        return np.sin(2 * np.pi * x)
    if case == 2:
        return np.sin(5 * np.pi * x)
    if case == 3:
        psi = np.zeros_like(x)
        set = (x > 1/5) & (x < 2/5)
        psi[set] = np.sin(5 * np.pi * x[set])
        return psi

def string_simulation(case, N=100, dt=0.001, T=1, c=1, L=1):
    dx = L / N
    a = c * dt / dx
    x = np.linspace(0, L, N + 1)

    psi_prev = initial_condition(x, case)
    psi_prev[0] = 0
    psi_prev[-1] = 0

    psi_curr = np.zeros(N+1)
    for i in range(1, N):
        psi_curr[i] = psi_prev[i] + 0.5 * a**2 * (psi_prev[i+1] - 2*psi_prev[i] + psi_prev[i-1])
    psi_curr[0] = 0
    psi_curr[-1] = 0

    nt = int(T / dt)

    plot_times = np.linspace(0.1, 1, 10)

    plt.plot(x, psi_prev, label=f"t={0.0}")

    data = []
    data.append(psi_prev.copy())

    for n in range(1, nt + 1):
        psi_next = np.zeros(N+1)

        for i in range(1, N):
            psi_next[i] = 2*psi_curr[i] - psi_prev[i] + a**2 * (psi_curr[i+1] - 2*psi_curr[i] + psi_curr[i-1])

        psi_next[0] = 0
        psi_next[-1] = 0

        if n*dt in plot_times:
            plt.plot(x, psi_curr, label=f"t={n*dt:.1f}")

        data.append(psi_curr.copy())

        psi_prev = psi_curr
        psi_curr = psi_next

    plt.title(f"Initial Condition {case}")
    plt.xlabel("x")
    plt.ylabel("Psi(x,t)")
    plt.legend()
    plt.show()

    return x, data


cases = [1, 2, 3]

for i in cases:
    x, data = string_simulation(i)

    fig, ax = plt.subplots()
    line, = ax.plot(x, data[0])
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("Psi(x,t)")
    ax.set_title(f"Simulation Initial Condition {i}")

    def update(frame):
        line.set_ydata(data[frame])
        return line,

    anim = animation.FuncAnimation(fig, update, frames=len(data), interval=50, blit=True)

    plt.show()