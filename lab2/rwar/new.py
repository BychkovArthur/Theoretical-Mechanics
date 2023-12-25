import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math



def Rot2D(X, Y, Alpha):
    RX = X*np.cos(Alpha) - Y*np.sin(Alpha)
    RY = X*np.sin(Alpha) + Y*np.cos(Alpha)
    return RX, RY


step = 1000
t = np.linspace(0, 30, step)

s = np.sin(5 * t)
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
gruz = ax1.plot(Ax[0], Ay[0], marker='o', c='r')[0]


K = 19
Sh = 0.4
b = 1/(K-2)
X_Spr = np.zeros(K)
Y_Spr = np.zeros(K)
X_Spr[0] = 0
Y_Spr[0] = 0
X_Spr[K-1] = 1
Y_Spr[K-1] = 0

spring = ax1.plot(X_Spr, Y_Spr)

for i in range(K-2):
    X_Spr[i+1] = b*((i+1) - 1/2)
    Y_Spr[i+1] = Sh*(-1)**i

# # ---
# # ПРУЖИНА
# # ---
# # получаю координаты пружины после поворота
# Spr_x_L_fi, Spr_y_fi = Rot2D(X_Spr, Y_Spr, -(math.pi/2 + abs(math.atan2( Spr_x[0], Spr_y[0]))))
# # задаю пружину уже после повторота, причём сразу перемещаю её в конечную позицию и растягиваю на длину
# WArrow, = ax.plot(Spr_x_L_fi + lenDE, (Spr_y_fi*length_Spr[0]) + l0)
# #крепёж для пружины
# ax.plot(2*a, l0, color='black', linewidth=5, marker='o')
# ax.plot([2*a-0.5, 2*a+0.5, 2*a, 2*a-0.5], [l0+0.7, l0+0.7, l0, l0+0.7], color='black', linewidth=2, )
# # ---
# ax.plot([-0.5, 0.5, 0, -0.5], [-0.5, -0.5, 0, -0.5], color='black', linewidth=2)
# ax.plot([-0.75, 0.75], [-0.5, -0.5], color='black', linewidth=3)

# Шаблон пружины
# Np = 20      # /\  /\  /\
#              #   \/  \/  \/
# H = 0.5
# Xp = np.linspace(0,1, 2 * Np + 1)
# Yp = 0.05 * np.sin(np.pi / 2 * np.arange(2 * Np + 1))

# Pruzh = ax1.plot((x0+s[0]) * Xp, Yp + 2 * R + H/2)[0]


def anima(i):
    O.set_data([Ox[i]], [Oy[i]])
    Circ.set_data(Ox[i] + R * np.cos(phiForCirc), Oy[i] + R * np.sin(phiForCirc))
    # Pruzh.set_data((s[i]) * Xp, Yp + 2*R + H/2)
    gruz.set_data(Ax[i], Ay[i])
    
anim = FuncAnimation(fig, anima, frames=100, interval=100)

plt.show()