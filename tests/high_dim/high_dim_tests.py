import numpy as np
import yroots as yr
from yroots.subdivision import solve
import scipy as sp
from time import time
from matplotlib import pyplot as plt

def residuals(funcs,roots):
    if len(roots) == 0:
        raise ValueError("No roots value error")
    else:
        Resid = list()
        for root in roots:
            for func in funcs:
                Resid.append(np.abs(func(*root)))

    return np.max(Resid), np.mean(Resid)

def test_2_0():
    fx = lambda x,y: 2*(x-1) + 200*(y-x**2)*(-2*x)
    fy = lambda x,y: 200*(y-x**2)

    a = [.5]*2
    b = [1.5]*2

    start = time ()
    roots = solve([fx,fy], a, b)
    end = time () - start

    num_roots = len(roots)
    max_resid, avg_resid = residuals([fx,fy],roots)

    return end, max_resid, avg_resid, num_roots



def test_3_0():
    f = lambda x,y,z : np.sin(x*z) + x*np.log(y+3) - x**2
    g = lambda x,y,z : np.cos(4*x*y) + np.exp(3*y/(x-2)) - 5
    h = lambda x,y,z : np.cos(2*y) - 3*z + 1/(x-8)

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time ()
    roots = solve([f,g,h], a, b)
    end = time () - start

    num_roots = len(roots)
    max_resid, avg_resid = residuals([f,g,h],roots)

    return end, max_resid, avg_resid, num_roots

def test_3_1():
    f = lambda x,y,z: np.cosh(4*x*y) + np.exp(z)- 5
    g = lambda x,y,z: x - np.log(1/(y+3))
    h = lambda x,y,z: x**2 -  z

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time ()
    roots = solve([f,g,h], a, b)
    end = time () - start

    num_roots = len(roots)
    max_resid, avg_resid = residuals([f,g,h],roots)

    return end, max_resid, avg_resid, num_roots

def test_3_2():
    f = lambda x,y,z: y**2-x**3
    g = lambda x,y,z: (y+.1)**3-(x-.1)**2
    h = lambda x,y,z: x**2 + y**2 + z**2 - 1

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    num_roots = len(roots)
    max_resid, avg_resid = residuals([f,g,h],roots)

    return end, max_resid, avg_resid, num_roots

def test_3_3():
    f = lambda x,y,z: 2*z**11 + 3*z**9 - 5*z**8 + 5*z**3 - 4*z**2 - 1
    g = lambda x,y,z: 2*y + 18*z**10 + 25*z**8 - 45*z**7 - 5*z**6 + 5*z**5 - 5*z**4 + 5*z**3 + 40*z**2 - 31*z - 6
    h = lambda x,y,z: 2*x - 2*z**9 - 5*z**7 + 5*z**6 - 5*z**5 + 5*z**4 - 5*z**3 + 5*z**2 + 1

    a = [-1,-1,-1]
    b = [1,1,1.2]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    num_roots = len(roots)
    max_resid, avg_resid = residuals([f,g,h],roots)

    return end, max_resid, avg_resid, num_roots

def test_3_4():
    f = lambda x,y,z: np.sin(4*(x + z) * np.exp(y))
    g = lambda x,y,z: np.cos(2*(z**3 + y + np.pi/7))
    h = lambda x,y,z: 1/(x+5) - y

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_5(): #rosenbrock 3D
    fx = lambda x,y,z: 2*(x-1) + 200*(y-x**2)*(-2*x)
    fy = lambda x,y,z: 200*(y-x**2) + 200*(z-y**2)*(-2*y) - 2*(1-y)
    fz = lambda x,y,z: 200*(z-y**2)

    a = [0,0,0]
    b = [2,2,5]

    start = time()
    roots = solve([fx,fy,fz], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([fx,fy,fz],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_6():
    f = lambda x,y,z: np.cos(10*x*y)
    g = lambda x,y,z: x + y**2
    h = lambda x,y,z: x + y - z

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_7():
    f = lambda x,y,z: np.exp(2*x)-3
    g = lambda x,y,z: -np.exp(x-2*y) + 11
    h = lambda x,y,z: x + y + 3*z

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_8():
    f1 = lambda x,y,z: 2*x / (x**2-4) - 2*x
    f2 = lambda x,y,z: 2*y / (y**2+4) - 2*y
    f3 = lambda x,y,z: 2*z / (z**2-4) - 2*z

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve([f1,f2,f3], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f1,f2,f3],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_9():
    f = lambda x,y,z: 2*x**2 / (x**4-4) - 2*x**2 + .5
    g = lambda x,y,z: 2*x**2*y / (y**2+4) - 2*y + 2*x*z
    h = lambda x,y,z: 2*z / (z**2-4) - 2*z

    a = [-1,-1,-1]
    b = [1,1,.8]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_10():
    f = lambda x,y,z: 144*((x*z)**4+y**4)-225*((x*z)**2+y**2) + 350*(x*z)**2*y**2+81
    g = lambda x,y,z: y-(x*z)**6
    h = lambda x,y,z: (x*z)+y-z

    a = [-1.-1,-2]
    b = [1,1,2]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_11():
    f = lambda x,y,z: x**2+y**2-.49**2
    g = lambda x,y,z: (x-.1)*(x*y - .2)
    h = lambda x,y,z: x**2 + y**2 - z**2

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_12():
    f = lambda x,y,z: (np.exp(y-z)**2-x**3)*((y-0.7)**2-(x-0.3)**3)*((np.exp(y-z)+0.2)**2-(x+0.8)**3)*((y+0.2)**2-(x-0.8)**3)
    g = lambda x,y,z: ((np.exp(y-z)+.4)**3-(x-.4)**2)*((np.exp(y-z)+.3)**3-(x-.3)**2)*((np.exp(y-z)-.5)**3-(x+.6)**2)*((np.exp(y-z)+0.3)**3-(2*x-0.8)**3)
    h = lambda x,y,z: x + y + z

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_13():
    f = lambda x,y,z: ((x*z-.3)**2+2*(np.log(y+1.2)+0.3)**2-1)
    g = lambda x,y,z: ((x-.49)**2+(y+.5)**2-1)*((x+0.5)**2+(y+0.5)**2-1)*((x-1)**2+(y-0.5)**2-1)
    h = lambda x,y,z: x**4 + (np.log(y+1.4)-.3) - z

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_14():
    f = lambda x,y,z: np.exp(x-2*x**2-y**2-z**2)*np.sin(10*(x+y+z+x*y**2))
    g = lambda x,y,z: np.exp(-x+2*y**2+x*y**2*z)*np.sin(10*(x-y-2*x*y**2))
    h = lambda x,y,z: np.exp(x**2*y**2)*np.cos(x-y+z)

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_15():
    f = lambda x,y,z: ((x-0.1)**2+2*(y*z-0.1)**2-1)*((x*y+0.3)**2+2*(z-0.2)**2-1)
    g = lambda x,y,z: (2*(x*z+0.1)**2+(y+0.1)**2-1)*(2*(z-0.3)**2+(y-0.15)**2-1)*((x-0.21)**2+2*(y-0.15)**2-1)
    h = lambda x,y,z: (2*(y+0.1)**2-(z+.15)**2-1)*(2*(x+0.3)**2+(z-.15)**2-1)

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_16():
    f = lambda x,y,z: np.sin(3*(x+y+z))
    g = lambda x,y,z: np.sin(3*(x+y-z))
    h = lambda x,y,z: np.sin(3*(x-y-z))

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_17():
    f = lambda x,y,z: x - 2 + 3*sp.special.erf(z)
    g = lambda x,y,z: np.sin(x*z)
    h = lambda x,y,z: x*y + y**2 - 1

    a=[-1,-1,-1]
    b=[1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

#end of problems from 3d_examples notebook

def test_3_18():
    #the one example from 3d yroots
    f1 = lambda x,y,z: 144*(x**4+y**4)-225*(x**2+y**2) + 350*x**2*y**2+81
    g1 = lambda x,y,z: y-x**6
    h = lambda x,y,z: x+y-z

    a = [-1,-1,-2]
    b = [1,1,2]

    start = time()
    roots = solve([f1,g1,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f1,g1,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

#3D testing-kate

def test_3_19():
    f = lambda x,y,z: x - y + .5
    g = lambda x,y,z: x + y
    h = lambda x,y,z: x - y - z

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_20():
    f = lambda x,y,z: y + x/2 + 1/10
    g = lambda x,y,z: y - 2.1*x + 2
    h = lambda x,y,z: x + y - z

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_21():
    f = lambda x,y,z: x
    g = lambda x,y,z: (x-.9999)**2 + y**2-1
    h = lambda x,y,z: x + y - z

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_22():
    f = lambda x,y,z: 2*y*np.cos(y**2)*np.cos(2*x)-np.cos(y)
    g = lambda x,y,z: 2*np.sin(y**2)*np.sin(2*x)-np.sin(x)
    h = lambda x,y,z: x + y - z

    a = [-4,-4,-4]
    b = [4,4,4]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_23():
    f = lambda x,y,z: 2*x*y*np.cos(y**2)*np.cos(2*x)-np.cos(x*y*z)
    g = lambda x,y,z: 2*np.sin(x*y**2)*np.sin(3*x*y)-np.sin(x*y*z)
    h = lambda x,y,z: 2*x*z*np.sin(y*z**2)*np.sin(2*z)-np.sin(x*y*z)

    a = [-2,-2,-2]
    b = [2,2,2]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_24():
    f = lambda x,y,z: np.sin(20*x+y)
    g = lambda x,y,z: np.cos(x**2+x*y)-.25
    h = lambda x,y,z: x + 4*y - 3*x

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_25():
    f = lambda x,y,z: (y - 2*x)*(y+0.5*x)
    g = lambda x,y,z: x*(x**2+y**2-1)
    h = lambda x,y,z: x + y - z

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_26():
    f = lambda x,y,z: (y - 2*x)*(y+.5*x)
    g = lambda x,y,z: (x-.0001)*(x**2+y**2-1)
    h = lambda x,y,z: x + y - z

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_27():
    f = lambda x,y,z: 25*x*y - 12
    g = lambda x,y,z: x**2 + y**2 - 1
    h = lambda x,y,z: x + y - z

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_28():
    f = lambda x,y,z: (x**2+y**2-1)*(x-1.1)
    g = lambda x,y,z: (25*x*y-12)*(x-1.1)
    h = lambda x,y,z: x + y - z

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_29():
    f = lambda x,y,z : np.sin(3*np.pi*x)*np.cos(x*y)
    g = lambda x,y,z : np.sin(3*np.pi*y)*np.cos(np.sin(x*y))
    h = lambda x,y,z : 3*x - 5*y + z

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_30():
    f = lambda x,y,z: np.sin(10*x-y/10)
    g = lambda x,y,z: np.cos(3*x*y)
    h = lambda x,y,z: x + y - z

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_31():
    f = lambda x,y,z: np.sin(10*x-y/10) + y
    g = lambda x,y,z: np.cos(10*y-x/10) - x
    h = lambda x,y,z: x + y - z

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_32():
    f = lambda x,y,z: x**2+y**2-.9**2
    g = lambda x,y,z: np.sin(x*y)
    h = lambda x,y,z: x + y - z

    a = [-1,-1,-1]
    b = [1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_33():
    f = lambda x,y,z: (x-1)*(np.cos(x*y**2)+2)
    g = lambda x,y,z: np.sin(8*np.pi*y)*(np.cos(x*y)+2)
    h = lambda x,y,z: x + y + z

    a = [-1,-1,-1]
    b = [1.000000001]*3

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_34():
    f = lambda x,y,z: (x-1)*(np.cos(x*y**2)+2)
    g = lambda x,y,z: np.sin(8*np.pi*y)*(np.cos(x*y)+2)
    h = lambda x,y,z: x + y + z

    a = [-1,-1,-1]
    b = [1,1.1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots
#end 3D testing-kate
def test_3_35():
    f = lambda x,y,z: x + 2*y - z/3
    g = lambda x,y,z: sp.stats.norm.cdf(x-40)
    h = lambda x,y,z: x**2

    a=[-1,-1,-1]
    b=[1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_36():
    f = lambda x,y,z: x + 2*y - z/3
    g = lambda x,y,z: sp.stats.norm.cdf(x)-.5
    h = lambda x,y,z: x**2

    a=[-1,-1,-1]
    b=[1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_37():
    f = lambda x,y,z: sp.stats.norm.pdf(y-14)
    g = lambda x,y,z: (2/(y+4))**(1/2) - np.cos(x)
    h = lambda x,y,z: y**2 + 2*z

    a=[-1,-1,-1]
    b=[1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_3_38():
    f = lambda x,y,z: sp.stats.norm.pdf(y*x - 10)
    g = lambda x,y,z: (2/(y+4))**(1/2) - np.cos(x)
    h = lambda x,y,z: y**2 + 2*z

    a=[-1,-1,-1]
    b=[1,1,1]

    start = time()
    roots = solve([f,g,h], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots
#end scipy functions

def test_4_0():
    f1 = lambda x1,x2,x3,x4 : np.sin(x1*x3) + x1*np.log(x2+3) - x1**2
    f2 = lambda x1,x2,x3,x4 : np.cos(4*x1*x2) + np.exp(3*x2/(x1-2)) - 5
    f3 = lambda x1,x2,x3,x4 : np.cos(2*x2) - 3*x3 + 1/(x1-8)
    f4 = lambda x1,x2,x3,x4 : x1 + x2 - x3 - x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f1,f2,f3,f4], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f1,f2,f3,f4],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_4_1():
    f = lambda x,y,z,x4: np.cosh(4*x*y) + np.exp(z)- 5
    g = lambda x,y,z,x4: x - np.log(1/(y+3))
    h = lambda x,y,z,x4: x**2 -  z
    f4 = lambda x,y,z,x4: x + y + z + x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h,f4],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_4_2():
    f = lambda x,y,z,x4: y**2-x**3
    g = lambda x,y,z,x4: (y+.1)**3-(x-.1)**2
    h = lambda x,y,z,x4: x**2 + y**2 + z**2 - 1
    f4 = lambda x,y,z,x4: x + y + z + x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h,f4],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_4_3():
    f = lambda x,y,z,x4: 2*z**11 + 3*z**9 - 5*z**8 + 5*z**3 - 4*z**2 - 1
    g = lambda x,y,z,x4: 2*y + 18*z**10 + 25*z**8 - 45*z**7 - 5*z**6 + 5*z**5 - 5*z**4 + 5*z**3 + 40*z**2 - 31*z - 6
    h = lambda x,y,z,x4: 2*x - 2*z**9 - 5*z**7 + 5*z**6 - 5*z**5 + 5*z**4 - 5*z**3 + 5*z**2 + 1
    f4 = lambda x,y,z,x4: x - y - z + x4

    a = [-1,-1,-1,-1]
    b = [1,1,1.2,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h,f4],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_4_4():
    f = lambda x,y,z,x4: np.sin(4*(x + z) * np.exp(y))
    g = lambda x,y,z,x4: np.cos(2*(z**3 + y + np.pi/7))
    h = lambda x,y,z,x4: 1/(x+5) - y
    f4 = lambda x,y,z,x4: x - y + z - x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h,f4],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_4_5():
    f = lambda x,y,z,x4: np.cos(10*x*y)
    g = lambda x,y,z,x4: x + y**2
    h = lambda x,y,z,x4: x + y - z
    f4 = lambda x,y,z,x4: x - y + z + x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h,f4],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_4_6():
    f = lambda x,y,z,x4: np.exp(2*x)-3
    g = lambda x,y,z,x4: -np.exp(x-2*y) + 11
    h = lambda x,y,z,x4: x + y + 3*z
    f4 = lambda x,y,z,x4: x + y + z + x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h,f4],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_4_7():
    f1 = lambda x,y,z,x4: 2*x / (x**2-4) - 2*x
    f2 = lambda x,y,z,x4: 2*y / (y**2+4) - 2*y
    f3 = lambda x,y,z,x4: 2*z / (z**2-4) - 2*z
    f4 = lambda x,y,z,x4: 2*x4 / (x4 **2 - 4) - 2*x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f1,f2,f3,f4], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f1,f2,f3,f4],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_4_8():
    f = lambda x,y,z,x4: 2*x**2 / (x**4-4) - 2*x**2 + .5
    g = lambda x,y,z,x4: 2*x**2*y / (y**2+4) - 2*y + 2*x*z
    h = lambda x,y,z,x4: 2*z / (z**2-4) - 2*z
    f4 = lambda x,y,z,x4:x + y + z + x4

    a = [-1,-1,-1,-1]
    b = [1,1,.8,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h,f4],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_4_9():
    f = lambda x,y,z,x4: 144*((x*z)**4+y**4)-225*((x*z)**2+y**2) + 350*(x*z)**2*y**2+81
    g = lambda x,y,z,x4: y-(x*z)**6
    h = lambda x,y,z,x4: (x*z)+y-z
    f4 = lambda x,y,z,x4:-x - y + z - x4

    a = [-1,-1,-2,-1]
    b = [1,1,2,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h,f4],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_4_10():
    f = lambda x,y,z,x4: x**2+y**2-.49**2
    g = lambda x,y,z,x4: (x-.1)*(x*y - .2)
    h = lambda x,y,z,x4: x**2 + y**2 - z**2
    f4 = lambda x,y,z,x4: -x + y + z + x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h,f4],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_4_11():
    f = lambda x,y,z,x4: (np.exp(y-z)**2-x**3)*((y-0.7)**2-(x-0.3)**3)*((np.exp(y-z)+0.2)**2-(x+0.8)**3)*((y+0.2)**2-(x-0.8)**3)
    g = lambda x,y,z,x4: ((np.exp(y-z)+.4)**3-(x-.4)**2)*((np.exp(y-z)+.3)**3-(x-.3)**2)*((np.exp(y-z)-.5)**3-(x+.6)**2)*((np.exp(y-z)+0.3)**3-(2*x-0.8)**3)
    h = lambda x,y,z,x4: x + y + z
    f4 = lambda x,y,z,x4: x + y + z + x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h,f4],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_4_12():
    f = lambda x,y,z,x4: ((x*z-.3)**2+2*(np.log(y+1.2)+0.3)**2-1)
    g = lambda x,y,z,x4: ((x-.49)**2+(y+.5)**2-1)*((x+0.5)**2+(y+0.5)**2-1)*((x-1)**2+(y-0.5)**2-1)
    h = lambda x,y,z,x4: x**4 + (np.log(y+1.4)-.3) - z
    f4 = lambda x,y,z,x4:x - y + z - x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h,f4],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_4_13():
    f = lambda x,y,z,x4: np.exp(x-2*x**2-y**2-z**2)*np.sin(10*(x+y+z+x*y**2))
    g = lambda x,y,z,x4: np.exp(-x+2*y**2+x*y**2*z)*np.sin(10*(x-y-2*x*y**2))
    h = lambda x,y,z,x4: np.exp(x**2*y**2)*np.cos(x-y+z)
    f4 = lambda x,y,z,x4: x - y + z - x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h,f4],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_4_14():
    f = lambda x,y,z,x4: ((x-0.1)**2+2*(y*z-0.1)**2-1)*((x*y+0.3)**2+2*(z-0.2)**2-1)
    g = lambda x,y,z,x4: (2*(x*z+0.1)**2+(y+0.1)**2-1)*(2*(z-0.3)**2+(y-0.15)**2-1)*((x-0.21)**2+2*(y-0.15)**2-1)
    h = lambda x,y,z,x4: (2*(y+0.1)**2-(z+.15)**2-1)*(2*(x+0.3)**2+(z-.15)**2-1)
    f4 = lambda x,y,z,x4: x - y + z - x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h,f4],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_4_15():
    f = lambda x,y,z,x4: np.sin(3*(x+y+z))
    g = lambda x,y,z,x4: np.sin(3*(x+y-z))
    h = lambda x,y,z,x4: np.sin(3*(x-y-z))
    f4 = lambda x,y,z,x4: -x + y - z + x4

    a = [-1,-1,-1,-1]
    b = [1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h,f4],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots

def test_4_16():
    f = lambda x,y,z,x4: x - 2 + 3*sp.special.erf(z)
    g = lambda x,y,z,x4: np.sin(x*z)
    h = lambda x,y,z,x4: x*y + y**2 - 1
    f4 = lambda x,y,z,x4: x + y - z + x4

    a=[-1,-1,-1,-1]
    b=[1,1,1,1]

    start = time()
    roots = solve([f,g,h,f4], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([f,g,h,f4],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots
#end 4d_examples-Copy1
def test_4_17(): #rosenbrock 4D
    fx1 = lambda x1,x2,x3,x4: 200*(x2-x1**2)*(-2*x1) + 2*(x1-1)
    fx2 = lambda x1,x2,x3,x4: 200*(x2-x1**2) + 200*(x3-x2**2)*(-2*x2) + 2*(x2-1)
    fx3 = lambda x1,x2,x3,x4: 200*(x3-x2**2) + 200*(x4-x3**2)*(-2*x3) + 2*(x3-1)
    fx4 = lambda x1,x2,x3,x4: 200*(x4-x3**2)

    a = [.5]*4
    b = [1.5]*4

    start = time()
    roots = solve([fx1,fx2,fx3,fx4], a, b)
    end = time() - start

    max_resid, avg_resid = residuals([fx1,fx2,fx3,fx4],roots)
    num_roots = len(roots)

    return end, max_resid, avg_resid, num_roots
#rosenbrock has a 5d, 6d example if we ever want to use it


def mins_seconds(timing):
    mins = timing/60
    mins_int = int(mins//1)
    secs = (mins - mins_int)*60
    return str(mins_int) +"m " + str(secs) + "s"



if __name__=="__main__":
    tests = [#test_2_0,
            # test_3_0,
            # test_3_1,
            # test_3_2,
            # test_3_3,
            # test_3_4,
            # test_3_5,
            # test_3_6,
            # test_3_7,
            # test_3_8,
            # test_3_9,
            # #test_3_10
            # test_3_11,
            # test_3_12,
            # test_3_13,
            # test_3_14,
            # test_3_15,
            # test_3_16,
            # test_3_17,
            # test_3_18,
            # test_3_19,
            # test_3_20,
            # test_3_21,
            # # test_3_22,
            # test_3_23,
            # #test_3_24,
            # test_3_25,
            # test_3_26,
            # #test_3_27,
            # #test_3_28,
            # test_3_29,
            # test_3_30,
            # test_3_31,
            # test_3_32,
#             test_3_33,
 #            test_3_34,
             #test_3_35,
             #test_3_36,
             #test_3_37,
             #test_3_38,
            test_4_0,
            test_4_1,
            test_4_2,
            test_4_3,
            test_4_4,
            test_4_5,
            test_4_6,
            test_4_7,
            test_4_8,
            test_4_9,
            test_4_10,
            test_4_11,
            test_4_12,
            test_4_13,
            test_4_14,
            test_4_15,
            test_4_16,
            test_4_17]

    test_names = [#"2_0",
                    # "3_0",
                    # "3_1",
                    # "3_2",
                    # "3_3",
                    # "3_4",
                    # "3_5",
                    # "3_6",
                    # "3_7",
                    # "3_8",
                    # "3_9",
                    # #"3_10"
                    # "3_11",
                    # "3_12",
                    # "3_13",
                    # "3_14",
                    # "3_15",
                    # "3_16",
                    # "3_17",
                    # "3_18",
                    # "3_19",
                    # "3_20",
                    # "3_21",
                    # "3_22",
                    # "3_23",
                    # #"3_24",
                    #"3_25",
                    #"3_26",
                    # #"3_27",
                    # #"3_28",
                    # "3_29",
                    # "3_30",
                    # "3_31",
                    # "3_32",
                    # "3_33",
                    # "3_34"]
                     #"3_35",
                     #"3_36",
                     #"3_37",
                     #"3_38",
                    "4_0",
                    "4_1",
                    "4_2",
                    "4_3",
                    "4_4",
                    "4_5",
                    "4_6",
                    "4_7",
                    "4_8",
                    "4_9",
                    "4_10",
                    "4_11",
                    "4_12",
                    "4_13",
                    "4_14",
                    "4_15",
                    "4_16",
                    "4_17"]

    times = []
    max_resids = []
    avg_resids = []
    all_num_roots = []

    i = 0
    for test in tests:
        print("=========================================================")
        print("Test", test_names[i])
        timing, max_resid, avg_resid, num_roots = test()
        print("=========================================================")
        print("Time to solve: " + mins_seconds(timing))
        print("Maximum residual: " + str(max_resid))
        print("Average residual: " + str(avg_resid))
        print("\n")
        times.append(timing)
        max_resids.append(max_resid)
        avg_resids.append(avg_resid)
        all_num_roots.append(num_roots)
        i = i + 1

       # np.save("Old_Checks_w_macaulay(4)_timings_high_dim", times, allow_pickle=True, fix_imports=True)
       # np.save("Old_Checks_w_macaulay(4)_max_resids_high_dim", max_resids, allow_pickle=True, fix_imports=True)
       # np.save("Old_Checks_w_macaulay(4)_avg_resids_high_dim", avg_resids, allow_pickle=True, fix_imports=True)
       # np.save("Old_Checks_w_macaulay(4)_num_roots_high_dim", all_num_roots, allow_pickle=True, fix_imports=True)



#compare timings, max residuals, average residuals, intervals checked?, number of roots found
#save lists as np arrays? extension .npy

