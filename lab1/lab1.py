import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


'''
    Матрица поворта. Для вращения точек
'''
def Rot2D(X, Y, Alpha):
    RX = X*np.cos(Alpha) - Y*np.sin(Alpha)
    RY = X*np.sin(Alpha) + Y*np.cos(Alpha)
    return RX, RY

sizeArrow = 0.7

'''
    Создаём последовательность данных для времени
    Здесь время будет идти от 0 до 10
    И всего точек на данном промежутке - 1000
'''
T = np.linspace(0, 10, 1000)

'''
    Символическая переменная - время
'''
t = sp.Symbol('t')

''' 
    Наш закон движения
    Задан параметрически через время
'''
r = 1 + sp.sin(5 * t)
phi = t + 0.3 * sp.sin(30 * t)

'''
    Переходим от полярных координат, к Декартовым
'''
x = r * sp.cos(phi)
y = r * sp.sin(phi)

'''
    Считаем проихводные и находим вектор скорости и находим его длину (длину вектора скорости)
'''
Vx = sp.diff(x, t)
Vy = sp.diff(y, t)
v = (Vx ** 2 + Vy ** 2) ** 0.5


'''
    Дифференцируем скорость, находим ускорение и его длину (длину вектора ускорения)
'''
Wx = sp.diff(Vx, t)
Wy = sp.diff(Vy, t)
w = (Wx ** 2 + Wy ** 2) ** 0.5

'''
    Нормируем ускорение
'''
WxN = Wx / w
WyN = Wy / w


'''
    Модуль тангенсиального ускорения
'''
Wtan = sp.diff(v, t)
'''
    Модуль нормальной состовляющей ускорения
'''
Wnor = (w ** 2 - Wtan ** 2) ** 0.5
'''
    Радиус кривизны
'''
ro = v ** 2 / Wnor


'''
    Радиус кривизны сонаправлен с нормальным ускорением
    Вектор полного ускорения - это сумма векторов нормального и тангенциального ускорений
    Следовательно, Вектор нормального ускорения - это разность векторов полного и тангенциального ускорений
    Но у нас нет вектора тангенциального ускорения - только его модуль

    Но он сонаправлен со скоростью - отнормируем вектор скорости, умножим на модуль тангенциального ускорения
    Нормируем вектор нормального ускорения, получим вектор n
    Зная направляющий вектор и радиус кривизны сможем отрисовать кривизну
'''

'''
    Нормируем каждую из составляющих вектора скорости
    Т.к. вектор скорости сонаправлен с тангенциальным ускорением => 
    Умножением на модуль тангенциального ускорения получаем составляющие тангенциального ускорения:
'''
WTanx = Vx / v * Wtan
WTany = Vy / v * Wtan

'''
    Т.к. вектор ускорения - сумма тангенциального и нормального, 
    выразим нормальное (сейчас оно не нормированно)
'''
NotNormNx = Wx - WTanx
NotNormNy = Wy - WTany
NotNormn = (NotNormNx ** 2 + NotNormNy ** 2) ** 0.5

'''
    Нормируем нормальное ускорение
'''
Nx = NotNormNx / NotNormn
Ny = NotNormNy / NotNormn

'''
    Получаем компоненты вектора кривизны
'''
Curvex = Nx * ro
Curvey = Ny * ro


'''
    Массивы из нулей для всех наших величин
    Здесь создастся значение (ноль) для каждого элемента времени
'''
X = np.zeros_like(T)
Y = np.zeros_like(T)


VX = np.zeros_like(T)
VY = np.zeros_like(T)

WX = np.zeros_like(T)
WY = np.zeros_like(T)


RO = np.zeros_like(T)
CurveX = np.zeros_like(T)
CurveY = np.zeros_like(T)

'''
    Заполняем все значения
    sp.Subs(a, b, c) подставляет в формулу переменную b формулы a значение c
'''
for i in np.arange(len(T)):
    X[i] = sp.Subs(x, t, T[i])
    Y[i] = sp.Subs(y, t, T[i])
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])
    WX[i] = sp.Subs(WxN, t, T[i])
    WY[i] = sp.Subs(WyN, t, T[i])

    RO[i] = sp.Subs(ro, t, T[i])
    CurveX[i] = sp.Subs(Curvex, t, T[i])
    CurveY[i] = sp.Subs(Curvey, t, T[i])
    

fig = plt.figure() # создаю поле для отрисовки
ax1 = fig.add_subplot(1, 1, 1) # создаю ячейку на области для отрисовки в первую стркоу в первый столбец и единственную отрисовку
ax1.plot(X, Y) # Строим график по точкам из массивов X, Y
ax1.set_ylim([-6, 6]) # Пределы для осей
ax1.set_xlim([-6, 6])
ax1.set_aspect('equal') # Чтобы оси были равными

P, = ax1.plot(X[0], Y[0], marker='o') # Точка в начальной позиции
VLine, = ax1.plot([X[0], X[0]+VX[0]], [Y[0], Y[0]+VY[0]], 'r') # Вектор скорости (красный)
SLine, = ax1.plot([0, X[0]], [0, Y[0]], 'g') # Вектор перемещения (зеленый)
WLine, = ax1.plot([X[0], X[0]+WX[0]], [Y[0], Y[0]+WY[0]], 'b') # Вектор ускорения (синий)
CurveLine, = ax1.plot([X[0], X[0]+CurveX[0]], [Y[0], Y[0]+CurveY[0]], 'c') # Вектор радиуса кривизны (бирюзовый)


'''
    Рисуем стрелочки для векторов
'''

ArrowX = np.array([-0.2*sizeArrow, 0, -0.2*sizeArrow])
ArrowY = np.array([0.1*sizeArrow, 0, -0.1*sizeArrow])

RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))
VArrow, = ax1.plot(RArrowX+X[0]+VX[0], RArrowY+Y[0]+VY[0], 'r')

ArrowX_S = np.array([-0.2*sizeArrow, 0, -0.2*sizeArrow])
ArrowY_S = np.array([0.1*sizeArrow, 0, -0.1*sizeArrow])

RArrowX_S, RArrowY_S = Rot2D(ArrowX_S, ArrowY_S, math.atan2(Y[0], X[0]))
SArrow, = ax1.plot(RArrowX+X[0], RArrowY+Y[0], 'g')

ArrowX_W = np.array([-0.2*sizeArrow, 0, -0.2*sizeArrow])
ArrowY_W = np.array([0.1*sizeArrow, 0, -0.1*sizeArrow])

RArrowX_W, RArrowY_W = Rot2D(ArrowX_W, ArrowY_W, math.atan2(WY[0], WX[0]))
WArrow, = ax1.plot(RArrowX+X[0]+WX[0], RArrowY+Y[0]+WY[0], 'b')

ArrowX_CurveLine = np.array([-0.2*sizeArrow, 0, -0.2*sizeArrow])
ArrowY_CurveLine = np.array([0.1*sizeArrow, 0, -0.1*sizeArrow])

RArrowX_CurveLine, RArrowY_CurveLine = Rot2D(ArrowX_W, ArrowY_W, math.atan2(CurveY[0], CurveX[0]))
CurveLineArrow, = ax1.plot(RArrowX+X[0]+CurveX[0], RArrowY+Y[0]+CurveY[0], 'c')

'''
    Функция, производящая анимацию
'''
def anima(i):
    P.set_data(X[i], Y[i]) # Положение текущий точки
    VLine.set_data([X[i], X[i]+VX[i]], [Y[i], Y[i]+VY[i]])
    SLine.set_data([0, X[i]], [0, Y[i]])
    WLine.set_data([X[i], X[i]+WX[i]], [Y[i], Y[i]+WY[i]])
    CurveLine.set_data([X[i], X[i]+CurveX[i]], [Y[i], Y[i]+CurveY[i]])
    
    '''
        Рисуем стрелочки
    '''
    
    RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[i], VX[i]))
    VArrow.set_data(RArrowX+X[i]+VX[i], RArrowY+Y[i]+VY[i])

    RArrowX_S, RArrowY_S = Rot2D(ArrowX, ArrowY, math.atan2(Y[i], X[i]))
    SArrow.set_data(RArrowX_S+X[i], RArrowY_S+Y[i])

    RArrowX_W, RArrowY_W = Rot2D(ArrowX, ArrowY, math.atan2(WY[i], WX[i]))
    WArrow.set_data(RArrowX_W+X[i]+WX[i], RArrowY_W+Y[i]+WY[i])
    
    RArrowX_CurveLine, RArrowY_CurveLine = Rot2D(ArrowX, ArrowY, math.atan2(WY[i], WX[i]))
    CurveLineArrow.set_data(RArrowX_CurveLine+X[i]+CurveX[i], RArrowY_CurveLine+Y[i]+CurveY[i])
    return P

'''
    Запускаем анимацию
'''
anim = FuncAnimation(fig, anima, frames=1000, interval=100, repeat=True)

'''
    Здесь окно графика остается открытым и программа продолжает выполнение
'''
plt.show()
