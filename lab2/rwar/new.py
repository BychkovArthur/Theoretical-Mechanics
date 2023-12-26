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
'''
    Точки для пружины внутри обруча
'''
SpringDot1_x = Ox + np.cos(phi)
SpringDot1_y = Oy + np.sin(phi)
SpringDot2_x = Ox - np.cos(phi)
SpringDot2_y = Oy - np.sin(phi)


fig = plt.figure()
ax1 = fig.add_subplot()
ax1.axis('equal')
plt.gca().set_adjustable("box")
ax1.set(xlim=[0, 20], ylim=[0, 5])

'''
    Центр обруча
'''
O = ax1.plot(Ox[0], Oy[0], marker='o', c='b')[0]

phiForCirc = np.linspace(0, 2 * math.pi, 100)
Circ = ax1.plot(Ox[0] + R * np.cos(phiForCirc), Oy[0] + R * np.sin(phiForCirc))[0]
line = ax1.plot([3.5, 3.5], [0, 5])[0]

leftSpringDot = ax1.plot(3.5, R, marker='o', c='b')[0]
gruz = ax1.plot(Ax[0], Ay[0], marker='o', c='r')[0]
SpringDot1 = ax1.plot(SpringDot1_x[0], SpringDot1_y[0], marker='o', c='g')[0]
SpringDot2 = ax1.plot(SpringDot2_x[0], SpringDot2_y[0], marker='o', c='g')[0]


'''
    Пружина
'''
Np = 20
Xp = np.linspace(0, 1, 2 * Np + 1)
Yp = 0.5 * np.sin(np.pi / 2 * np.arange(2 * Np + 1))
Spring = ax1.plot((Ox[0] - 3.5) * Xp + 3.5, Yp + R, c='b')[0]

Np2 = 5
Xp2 = np.linspace(0, 1/10, 2 * Np2 + 1)
Yp2 = 0.1 * np.sin(np.pi / 2 * np.arange(2 * Np2 + 1))
Spring1 = ax1.plot((SpringDot1_x[0]) * Xp2 + SpringDot1_x[0], Yp2 + SpringDot1_y[0], c='g')[0]


def anima(i):
    O.set_data([Ox[i]], [Oy[i]])
    Circ.set_data(Ox[i] + R * np.cos(phiForCirc), Oy[i] + R * np.sin(phiForCirc))
    gruz.set_data(Ax[i], Ay[i])
    SpringDot1.set_data(SpringDot1_x[i], SpringDot1_y[i])
    SpringDot2.set_data(SpringDot2_x[i], SpringDot2_y[i])
    
    Spring.set_data((Ox[i] - 3.5) * Xp + 3.5, Yp + R)
    Spring1.set_data((SpringDot1_x[i]) * Xp2 + SpringDot1_x[i], Yp2 + SpringDot1_y[i])
    
    
anim = FuncAnimation(fig, anima, frames=100, interval=100)

plt.show()