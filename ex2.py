import numpy as np
import matplotlib.pyplot as plt
from math import erfc, sqrt
import matplotlib.animation as animation

def diffusion(N, dt, T, D, save_times):

    dx = 1 / N
    coeff = dt * D / (dx * dx)

    if 4 * coeff > 1:
        print("The stability condition is violated: use a lower time step")

    n_steps = int(T / dt)

    saved_solutions = {}
    save_steps = []
    for t in save_times:
        save_steps.append(round(t / dt))


    # Initialization

    c_old = np.zeros([N + 1, N + 1])
    c_new = np.zeros([N + 1, N + 1])

    c_old[0, :] = 0
    c_old[N, :] = 1

    c_old[:, N] = c_old[:, 0]

    if 0 in save_steps:
        saved_solutions[0] = c_old.copy()

    c_animation = []
    c_animation.append(c_old.copy())

    # time simulation
    for k in range(0, n_steps):

        # internal nodes
        for j in range(1, N):
            for i in range(0, N):

                c_center = c_old[j, i]

                if i == 0:
                    c_left = c_old[j, N - 1]
                    c_right = c_old[j, i + 1]
                elif i == N - 1:
                    c_left = c_old[j, i - 1]
                    c_right = c_old[j, 0]
                else:
                    c_left = c_old[j, i - 1]
                    c_right = c_old[j, i + 1]

                c_down = c_old[j - 1, i]
                c_up = c_old[j + 1, i]

                increment = c_left + c_right + c_down + c_up - 4 * c_center
                c_new[j, i] = c_center + coeff * increment

        # boundary in Y
        for i in range(0, N + 1):
            c_new[0, i] = 0
            c_new[N, i] = 1

        # boundary in X
        c_new[:, N] = c_new[:, 0]

        c_old[:] = c_new

        c_animation.append(c_old.copy())

        step_now = k + 1
        if step_now in save_steps:
            saved_solutions[step_now * dt] = c_old.copy()

    return c_old, dx, saved_solutions, c_animation


def analytic_solution(x, t, D, n_sum):
    dim = np.size(x)
    out = np.zeros(dim)
    denom = 2 * sqrt(D * t)
    out = []
    for i in x:
        s = 0
        for j in range(n_sum):
            a = (1 - i + 2*j) / denom
            b = (1 + i + 2*j) / denom
            s += erfc(a) - erfc(b)
        out = np.append(out, s)
    return out

# params
N = 5
D = 1
dx = 1 / N
dt = 0.0005

times = [0, 0.001, 0.01, 0.1, 1]
T = max(times)

c_final, dx, saved_solutions, c_animation = diffusion(N, dt, T, D, times)

x = np.linspace(0, 1, N+1)
n_sum = 500

plt.figure()
for t, c in saved_solutions.items():
    c_simulation = c[:, 0]  # depends only on y
    plt.plot(x, c_simulation, linestyle=':', label=f"simulation t={t:.3g}")

    if t > 0:
        c_analytic = analytic_solution(x, t, D, n_sum)
        plt.plot(x, c_analytic, linestyle='-', label=f"analytic t={t:.3g}")
    else:
        # the analytic is not defined at t=0 because denom(t=0)=0
        pass

plt.xlabel("y")
plt.ylabel("c(y,t)")
plt.legend()
plt.grid(True)
plt.show()


fig, ax = plt.subplots()
im = ax.imshow(c_animation[0], origin='lower', extent=[0,1,0,1])
fig.colorbar(im, ax=ax)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Diffusion Animation")

def update(frame):
    im.set_data(c_animation[frame])
    return im,

anim = animation.FuncAnimation(fig, update, frames=len(c_animation), interval=50, blit=True)

plt.show()