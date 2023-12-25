import numpy as np
from math import pi
import matplotlib
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import sympy as sp
import math

t_fin = 20
steps = 1001
T = np.linspace(0, t_fin, steps)
angles = np.linspace(0, 2 * pi, 360)

t = sp.Symbol('t')

phi = sp.sin(t)
s = sp.cos(t)


box_w = 0.4
box_h = 0.2
def spring(k, h, w):
    x = np.linspace(0, h, 100)
    return np.array([x, np.sin(2 * math.pi / (h / k) * x) * w])

box_x_tmp = np.array([-box_h / 2, -box_h / 2, box_h / 2, box_h / 2, -box_h / 2])
box_y_tmp = np.array([-box_w / 2, box_w / 2, box_w / 2, -box_w / 2, -box_w / 2])

# F_friction = np.zeros(1001)
# N = np.zeros(1001)

ring_dots_x = np.zeros([1001, 360])
ring_dots_y = np.zeros([1001, 360])

box_dots_x = np.zeros([1001, 5])
box_dots_y = np.zeros([1001, 5])

spring_a_x = np.zeros([1001, 100])
spring_a_y = np.zeros([1001, 100])
spring_b_x = np.zeros([1001, 100])
spring_b_y = np.zeros([1001, 100])

spring_c_x = np.zeros([1001, 100])
spring_c_y = np.zeros([1001, 100])

R = 1
x0 = 4

for i in range(1001):
    # F_friction[i] = (m1 + m2) * R * ddphi[i] - m2 * (dds[i] - s[i] * dphi[i]**2) * np.cos(phi[i]) + m2*(2*ds[i]*dphi[i] + s[i]*ddphi[i]) * np.sin(phi[i]) + c1*R*phi[i]

    # N[i] = m2*((dds[i] - s[i]*(dphi[i]**2))*np.sin(phi[i])+(2*ds[i]*dphi[i] + s[i]*ddphi[i])*np.cos(phi[i])) + (m1+m2)*g

    ring_x = x0 + phi[i] * R
    ring_y = R

    ring_dots_x[i] = np.cos(phi[i]) * R * np.cos(angles) + np.sin(phi[i]) * R * np.sin(angles) + ring_x
    ring_dots_y[i] = - np.sin(phi[i]) * R * np.cos(angles) + np.cos(phi[i]) * R * np.sin(angles) + ring_y

    bx = box_x_tmp - s[i]
    by = box_y_tmp
    box_dots_x[i] = np.cos(phi[i]) * bx + np.sin(phi[i]) * by + ring_x
    box_dots_y[i] = - np.sin(phi[i]) * bx + np.cos(phi[i]) * by + ring_y

    spring_a_x[i] = spring(5, ring_x, 0.2)[0]
    spring_a_y[i] = spring(5, ring_x, 0.2)[1] + ring_y

    b_x = R - spring(10, R + s[i] - box_h / 2, 0.16)[0]
    b_y = spring(10, R - s[i], 0.16)[1]
    spring_b_x[i] = np.cos(phi[i]) * b_x + np.sin(phi[i]) * b_y + ring_x
    spring_b_y[i] = -np.sin(phi[i]) * b_x + np.cos(phi[i]) * b_y + ring_y

    c_x = spring(10, R - s[i] - box_h / 2, 0.16)[0] - R
    c_y = spring(10, R - s[i], 0.16)[1]
    spring_c_x[i] = np.cos(phi[i]) * c_x + np.sin(phi[i]) * c_y + ring_x
    spring_c_y[i] = -np.sin(phi[i]) * c_x + np.cos(phi[i]) * c_y + ring_y

# fig_for_graphs = plt.figure(figsize=[13, 7])

# ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 1)
# ax_for_graphs.plot(t, F_friction, color='black')
# ax_for_graphs.set_title("F(t)")
# ax_for_graphs.set(xlim=[0, t_fin])
# ax_for_graphs.grid(True)

# ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 2)
# ax_for_graphs.plot(t, N, color='black')
# ax_for_graphs.set_title("N(t)")
# ax_for_graphs.set(xlim=[0, t_fin])
# ax_for_graphs.grid(True)

# ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 3)
# ax_for_graphs.plot(t, s, color='blue')
# ax_for_graphs.set_title("s(t)")
# ax_for_graphs.set(xlim=[0, t_fin])
# ax_for_graphs.grid(True)

# ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 4)
# ax_for_graphs.plot(t, phi, color='red')
# ax_for_graphs.set_title('phi(t)')
# ax_for_graphs.set(xlim=[0, t_fin])
# ax_for_graphs.grid(True)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.axis("equal")

surface = ax.plot([0, 0, 8], [5, 0, 0], "black")
ring, = ax.plot(ring_dots_x[0], ring_dots_y[0], "black")
box, = ax.plot(box_dots_x[0], box_dots_y[0], "black")
spring_a, = ax.plot(spring_a_x[0], spring_a_y[0], "red")
spring_b, = ax.plot(spring_b_x[0], spring_b_y[0], "purple")
spring_c, = ax.plot(spring_c_x[0], spring_c_y[0], "brown")

def animate(i):
    ring.set_data(ring_dots_x[i], ring_dots_y[i])
    box.set_data(box_dots_x[i], box_dots_y[i])
    spring_a.set_data(spring_a_x[i], spring_a_y[i])
    spring_b.set_data(spring_b_x[i], spring_b_y[i])
    spring_c.set_data(spring_c_x[i], spring_c_y[i])

    return ring, box, spring_a, spring_b, spring_c

animation = FuncAnimation(fig, animate, frames=1000, interval=60)
plt.show()