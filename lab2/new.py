import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

step = 1000
t = np.linspace(0, 30, step)

s = np.sin(t)
phi = 5 * np.sin(2*t)

x0 = 10
R = 1 # Радиус обруча

Ox = x0 + R * phi
Oy = [R] * 1000

Ax = Ox + s * np.cos(phi)
Ay = Oy + s * np.sin(phi)

fig = plt.figure()
ax1 = fig.add_subplot()
ax1.axis('equal')
plt.gca().set_adjustable("box")
ax1.set(xlim=[0, 20], ylim=[0, 5])


O = ax1.plot(Ox[0], Oy[0], marker='o', c='b')[0]

phiForCirc = np.linspace(0, 2 * math.pi, 100)
Circ = ax1.plot(Ox[0] + R * np.cos(phiForCirc), Oy[0] + R * np.sin(phiForCirc))[0]
line = ax1.plot([3.5, 3.5], [0, 5])[0]

dot = ax1.plot(3.5, R, marker='o', c='g')[0]


# Шаблон пружины
Np = 20      # /\  /\  /\
             #   \/  \/  \/
H = 0.5
Xp = np.linspace(0,1, 2 * Np + 1)
Yp = 0.05 * np.sin(np.pi / 2 * np.arange(2 * Np + 1))

Pruzh = ax1.plot((x0+s[0]) * Xp, Yp + 2 * R + H/2)[0]


def anima(i):
    O.set_data([Ox[i]], [Oy[i]])
    Circ.set_data(Ox[i] + R * np.cos(phiForCirc), Oy[i] + R * np.sin(phiForCirc))
    Pruzh.set_data((s[i]) * Xp, Yp + 2*R + H/2)
    
anim = FuncAnimation(fig, anima, frames=100, interval=100)

plt.show()