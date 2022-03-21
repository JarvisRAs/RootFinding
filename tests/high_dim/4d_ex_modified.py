import numpy as np
import yroots as yr
import scipy as sp
import matplotlib
from yroots.subdivision import solve
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time

def residuals(f,g,h,f4,roots,t):
    Resid = list()
    Root = list()
    for i in range(len(roots)):
        for j in range(4):
            Root.append(roots[i,j])
        Resid.append(np.abs(f(Root[0],Root[1],Root[2],Root[3])))
        Resid.append(np.abs(g(Root[0],Root[1],Root[2],Root[3])))
        Resid.append(np.abs(h(Root[0],Root[1],Root[2],Root[3])))
        Resid.append(np.abs(f4(Root[0],Root[1],Root[2],Root[3])))
        Root = []

    hours = int(t // 3600)
    minutes = int((t%3600) // 60)
    seconds = int((t%3600)%60 // 1)
    msecs = int(np.round((t % 1) * 1000,0))
    print("time elapsed: ",hours,"hours,", minutes,"minutes,",seconds, "seconds,",msecs, "milliseconds")
    print("Residuals: ", Resid, "\n")
    print("Max Residual: ", np.amax(Resid))
    return np.amax(Resid),t
'''
def plot_resids(residuals):
    plt.scatter([i+1 for i in range(18)],residuals)
    plt.ylim(1e-20,1e-7)
    plt.xticks(range(1, 19, 2))
    plt.yscale('log')
    plt.axhline(y=2.22044604925031e-13,c='r')
    plt.xlabel('example #')
    plt.ylabel('max residual')
    plt.title('max Residuals for 3d examples (log scale)')
    plt.show()
    return


def ex1():
    print("====================== ex 1 ======================")
    return 2.6645352591003757e-15

def ex2():
    print("====================== ex 2 ======================")
    return 3.552713678800501e-15

def ex3():
    print("====================== ex 3 ======================")
    return 4.440892098500626e-16

def ex4():
    print("====================== ex 4 ======================")
    return residuals(f,g,h,f4,roots,t)

def ex5():
    f = lambda x,y,z,x4: np.sin(4*(x + z) * np.exp(y))
    g = lambda x,y,z,x4: np.cos(2*(z**3 + y + np.pi/7))
    h = lambda x,y,z,x4: 1/(x+5) - y
    f4 = lambda x,y,z,x4: x - y + z - x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    t = time() - start
    print("====================== ex 5 ======================")
    return residuals(f,g,h,f4,roots,t)

#returns known residual of Rosenbrock in 4d
def ex6():
    print("====================== ex 6 ======================")
    return 1.11071152275615E-10

'''

def ex7():
    f = lambda x,y,z,x4: np.cos(10*x*y)
    g = lambda x,y,z,x4: x + y**2
    h = lambda x,y,z,x4: x + y - z
    f4 = lambda x,y,z,x4: x - y + z + x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    t = time() - start
    print("====================== ex 7 ======================")
    return residuals(f,g,h,f4,roots,t)

def ex8():
    f = lambda x,y,z,x4: np.exp(2*x)-3
    g = lambda x,y,z,x4: -np.exp(x-2*y) + 11
    h = lambda x,y,z,x4: x + y + 3*z
    f4 = lambda x,y,z,x4: x + y + z + x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = yr.solve([f,g,h,f4], a, b)
    t = time() - start
    print("====================== ex 8 ======================")
    return residuals(f,g,h,f4,roots,t)

def ex9():
    f1 = lambda x,y,z,x4: 2*x / (x**2-4) - 2*x
    f2 = lambda x,y,z,x4: 2*y / (y**2+4) - 2*y
    f3 = lambda x,y,z,x4: 2*z / (z**2-4) - 2*z
    f4 = lambda x,y,z,x4: 2*x4 / (x4 **2 - 4) - 2*x4

    start = time()
    roots = yr.solve([f1,f2,f3],[-1,-1,-1,-1],[1,1,1,1])
    t = time() - start
    print("====================== ex 9 ======================")
    return residuals(f,g,h,f4,roots,t)

def ex10():
    f = lambda x,y,z,x4: 2*x**2 / (x**4-4) - 2*x**2 + .5
    g = lambda x,y,z,x4: 2*x**2*y / (y**2+4) - 2*y + 2*x*z
    h = lambda x,y,z,x4: 2*z / (z**2-4) - 2*z
    f4 = lambda x,y,z,x4:x + y + z + x4

    start = time()
    roots = yr.solve([f1,f2,f3],[-1,-1,-1,-1],[1,1,.8,1])
    t = time() - start
    print("====================== ex 10 ======================")
    return residuals(f,g,h,f4,roots,t)

def ex11():
    f = lambda x,y,z,x4: 144*((x*z)**4+y**4)-225*((x*z)**2+y**2) + 350*(x*z)**2*y**2+81
    g = lambda x,y,z,x4: y-(x*z)**6
    h = lambda x,y,z,x4: (x*z)+y-z
    f4 = lambda x,y,z,x4:-x - y + z - x4

    start = time()
    roots = yr.solve([f,g,h,f4],[-1,-1,-2,-1],[1,1,2,1])
    t = time() - start
    print("====================== ex 11 ======================")
    return residuals(f,g,h,f4,roots,t)

def ex12():
    f = lambda x,y,z,x4: x**2+y**2-.49**2
    g = lambda x,y,z,x4: (x-.1)*(x*y - .2)
    h = lambda x,y,z,x4: x**2 + y**2 - z**2
    f4 = lambda x,y,z,x4: -x + y + z + x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    t = time() - start
    print("====================== ex 12 ======================")
    return residuals(f,g,h,f4,roots,t)

def ex13():
    f = lambda x,y,z,x4: (np.exp(y-z)**2-x**3)*((y-0.7)**2-(x-0.3)**3)*((np.exp(y-z)+0.2)**2-(x+0.8)**3)*((y+0.2)**2-(x-0.8)**3)
    g = lambda x,y,z,x4: ((np.exp(y-z)+.4)**3-(x-.4)**2)*((np.exp(y-z)+.3)**3-(x-.3)**2)*((np.exp(y-z)-.5)**3-(x+.6)**2)*((np.exp(y-z)+0.3)**3-(2*x-0.8)**3)
    h = lambda x,y,z,x4: x + y + z
    f4 = lambda x,y,z,x4: x + y + z + x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    t = time() - start
    print("====================== ex 13 ======================")
    return residuals(f,g,h,f4,roots,t)

def ex14():
    f = lambda x,y,z,x4: ((x*z-.3)**2+2*(np.log(y+1.2)+0.3)**2-1)
    g = lambda x,y,z,x4: ((x-.49)**2+(y+.5)**2-1)*((x+0.5)**2+(y+0.5)**2-1)*((x-1)**2+(y-0.5)**2-1)
    h = lambda x,y,z,x4: x**4 + (np.log(y+1.4)-.3) - z
    f4 = lambda x,y,z,x4:x - y + z - x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    t = time() - start
    print("====================== ex 14 ======================")
    return residuals(f,g,h,f4,roots,t)

def ex15():
    f = lambda x,y,z,x4: np.exp(x-2*x**2-y**2-z**2)*np.sin(10*(x+y+z+x*y**2))
    g = lambda x,y,z,x4: np.exp(-x+2*y**2+x*y**2*z)*np.sin(10*(x-y-2*x*y**2))
    h = lambda x,y,z,x4: np.exp(x**2*y**2)*np.cos(x-y+z)
    f4 = lambda x,y,z,x4: x - y + z - x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    t = time() - start
    print("====================== ex 15 ======================")
    return residuals(f,g,h,f4,roots,t)

def ex16():
    f = lambda x,y,z,x4: ((x-0.1)**2+2*(y*z-0.1)**2-1)*((x*y+0.3)**2+2*(z-0.2)**2-1)
    g = lambda x,y,z,x4: (2*(x*z+0.1)**2+(y+0.1)**2-1)*(2*(z-0.3)**2+(y-0.15)**2-1)*((x-0.21)**2+2*(y-0.15)**2-1)
    h = lambda x,y,z,x4: (2*(y+0.1)**2-(z+.15)**2-1)*(2*(x+0.3)**2+(z-.15)**2-1)
    f4 = lambda x,y,z,x4: x - y + z - x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    t = time() - start
    print("====================== ex 16 ======================")
    return residuals(f,g,h,f4,roots,t)

def ex17():
    f = lambda x,y,z,x4: np.sin(3*(x+y+z))
    g = lambda x,y,z,x4: np.sin(3*(x+y-z))
    h = lambda x,y,z,x4: np.sin(3*(x-y-z))
    f4 = lambda x,y,z,x4: -x + y - z + x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    t = time() - start
    print("====================== ex 17 ======================")
    return residuals(f,g,h,f4,roots,t)

def ex18():
    f = lambda x,y,z,x4: x - 2 + 3*sp.special.erf(z)
    g = lambda x,y,z,x4: np.sin(x*z)
    h = lambda x,y,z,x4: x*y + y**2 - 1
    f4 = lambda x,y,z,x4: x + y - z + x4

    a=[-1,-1,-1,-1]
    b=[1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4],a,b)
    t = time() - start
    print("====================== ex 18 ======================")
    return residuals(f,g,h,f4,roots,t)

if __name__ == "__main__":
    resids = []
    times = []
    for test in [ex7(),ex8(),ex9(),ex10(),ex11(),ex12(),ex13(),ex14(),ex15(),ex16(),ex17(),ex18()]:
        resid, t = tets()
        resids.append(resid)
        times.append(t)
    with open('4d_test_timings.npy','wb') as f:
        np.save(f,times)
    with open('4d_test_resids.npy', 'wb') as f:
        np.save(f,resids)
#    plot_resids(resids)
